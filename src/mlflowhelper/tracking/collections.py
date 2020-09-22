import collections.abc
import functools
import os
import pickle
import socket
import time
import typing
import uuid
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator

import cloudpickle
import mlflow
import mlflow.entities
import mlflow.exceptions
import mlflow.store.tracking
import mlflow.utils.mlflow_tags
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD

import mlflowhelper.tracking.artifactmanager

DICT_IDENTIFIER = "mlflow.tracking.collections.MlflowDict"


@dataclass
class MetaValue:
    value: typing.Any = None
    tags: typing.Union[list, dict, tuple] = None
    params: typing.Union[list, dict, tuple] = None
    metrics: typing.Union[list, dict, tuple] = None
    artifacts: typing.Union[list, dict, tuple] = None
    status: typing.Union[dict, tuple, str] = None
    update: bool = False


def cached(_func=None, *, cache, key):
    def decorator_cached(func):
        @functools.wraps(func)
        def wrapper_cached(*args, **kwargs):
            if key in cache:
                return cache[key]
            else:
                value = func(*args, **kwargs)
                cache[key] = value
            return value

        return wrapper_cached

    if _func is None:
        return decorator_cached
    else:
        return decorator_cached(_func)


class MlflowDict(collections.abc.MutableMapping):
    """
    Dict like class that persists elements to an Mlflow tracking server.
    WARNING: You need to explicitly set values again when you change them since the local cache is not automatically
    synchronized.
    """
    # TODO: Fix race conditions for synchronous access:
    #   We might be able to fix this by not deleting runs straight away.
    #   Instead we can manually select the most recent run when retrieving values.
    #   For clean-up we sweep through all runs every now and the and delete every run
    #   except the most recent ones for each key.
    #
    # TODO: Don't interpret runs with no value as valid keys.
    #   Or maybe it makes more sense to not interpret runs with `state!=FINISHED` as runs.

    class Pickler:
        """Abstract Pickler class which can be used for custom pickling."""

        @abstractmethod
        def dump(self, value, path):
            pass

        @abstractmethod
        def load(self, path):
            pass

    class DefaultPickler(Pickler):
        """Pickler using `pickle` as default"""

        def dump(self, value, path):
            with open(path, "wb") as f:
                pickle.dump(value, f)

        def load(self, path):
            with open(path, "rb") as f:
                return pickle.load(f)

    class CloudPickler(Pickler):
        """Pickler using `pickle` as default"""

        def dump(self, value, path):
            with open(path, "wb") as f:
                cloudpickle.dump(value, f)

        def load(self, path):
            with open(path, "rb") as f:
                return cloudpickle.load(f)

    class Logger:
        """Class for custom logging."""
        @abstractmethod
        def log(self, mlflow_client, artifact_manager, run, key, value) -> bool:
            """If you want to log a file, the artifact manager is used like this:
            >>> import matplotlib.pyplot as plt
            >>> with artifact_manager.managed_artifact("plot.png", dst_run_id=run.info.run_id) as a:
            >>>     plt.plot([1,2,3], [1,2,3])
            >>>     plt.savefig(a.get_path())
            """
            pass

    def __init__(
            self,
            mlflow_client: typing.Union[mlflow.tracking.client.MlflowClient, str] = None,
            mlflow_experiment_name="dict_db",
            mlflow_tag_dict_name="default",
            mlflow_tag_user=None,
            mlflow_tag_defaults=None,
            mlflow_tag_name_separator=": ",
            mlflow_tag_prefix="_mlflowdict",
            mlflow_custom_logger=None,
            mlflow_pickler: typing.Union[Pickler, str] = 'cloudpickle',
            sync_mode=None,
            local_value_cache=True,
            lazy_value_cache=True,
            read_only=False,
            only_load_finished_runs=False):
        """
        Dictionaries are identified by
        1) `mlflow_experiment_name` and 2) the `mlflow_tag_dict_name` in MLflow.
        The other tags are optional.

        :param mlflow_client:
            MlflowClient or str representing the tracking uri of MLflow.
        :param mlflow_experiment_name:
            Name of the MLflow experiment
        :param mlflow_tag_dict_name:
            Dictionary name set as `mlflow_tag_prefix`._name tag
        :param mlflow_tag_user:
            User name set as mlflow user tag
        :param mlflow_tag_defaults:
            Other default tags
        :param mlflow_tag_name_separator:
            The separator used to name runs which represent dict entries.
            Example: `{mlflow_tag_dict_name}{mlflow_tag_name_separator}{key_name}`
        :param mlflow_tag_prefix:
            Prefix for tags used by MlflowDict. You probably don't need to change this ever.
        :param mlflow_custom_logger:
            By default values are pickled and saved and that's it. If you want to log additional details about the value
            it can be done with a custom logger specified via this parameter.
            This can be a function or a class of type `MlflowDict.Logger`.
        :param mlflow_pickler:
            By default MlflowDict uses `pickle` to serialize values. You can provide a custom pickler here.
            See `MlflowDict.Pickler` for the interface. If set to `None`, no value will be logged and the result upon
            retrieving an element from the dict will be `None`.
        :param sync_mode:
            Specifies the synchronization mode of keys and items.

            WARNING:
            Any sync mode will trigger rather expensive database queries
            slowing down item retrieval and key listings considerably.
            This particularly heavily influences iterators over keys and items.

            Available modes:
            * None: no active synchronization.
            * "keys": will keep keys synchronized. This means that we do not cache keys,
                but update them every time this dict is accessed.
                This may slow down the dict operations a lot depending on
                access frequency, dict size and network latency.
            * "full": will keep keys (see "keys" mode) and locally cached values in sync.
                Caching values will only take effect if local value caching is activated (`local_value_cache=True`).
                If a value is in the local cache but the synchronized key has a newer timestamp
                then the value is updated.

        :param local_value_cache:
            Whether to keep a local cache of values.
        :param lazy_value_cache:
            Whether to download and cache all values on initialization (`lazy_cache=False`) or not.
        """

        self.lock_uuid = uuid.uuid4()

        # init client
        if mlflow_client is None:
            self.client = mlflow.tracking.MlflowClient()
        elif isinstance(mlflow_client, str):
            self.client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_client)
        else:
            self.client = mlflow_client

        # create experiment if it does not exist
        exp = self.client.get_experiment_by_name(mlflow_experiment_name)
        if exp is None:
            self.client.create_experiment(mlflow_experiment_name)
            exp = self.client.get_experiment_by_name(mlflow_experiment_name)
        self.experiment = exp

        # tags
        self.mlflow_tag_dict_name = mlflow_tag_dict_name
        self.mlflow_tag_user = mlflow_tag_user
        self.mlflow_tag_defaults = mlflow_tag_defaults
        self.mlflow_tag_prefix = mlflow_tag_prefix
        self.mlflow_tag_name_separator = mlflow_tag_name_separator

        self.mlflow_custom_logger = mlflow_custom_logger
        if mlflow_pickler == "pickle":
            self.mlflow_pickler = MlflowDict.DefaultPickler()
        if mlflow_pickler == "cloudpickle":
            self.mlflow_pickler = MlflowDict.CloudPickler()
        else:
            self.mlflow_pickler = mlflow_pickler

        self.only_load_finished_runs = only_load_finished_runs

        # sync
        self.sync_mode = sync_mode

        # caching
        self.local_cache = local_value_cache
        self.lazy_cache = lazy_value_cache

        self.local_values = dict()
        self.local_all_keys = self.update_keys()
        if not lazy_value_cache:
            self._init_values()

        # read only
        self.read_only = read_only

        # check whether dict exists
        if len(self.local_all_keys) > 0:
            warnings.warn("Dict already exists")

    def _init_values(self):
        self.local_values = {key: (value, int(run.data.tags[f"{self.mlflow_tag_prefix}._timestamp"]))
                             for key, value, run in self._get_items(return_runs=True)}

    def _log_value(self, key, value):

        timestamp = int(round(time.time() * 1000))

        # compile tags
        tags = dict()

        # required tags
        tags[f"{self.mlflow_tag_prefix}._class"] = DICT_IDENTIFIER
        tags[f"{self.mlflow_tag_prefix}._name"] = self.mlflow_tag_dict_name
        tags[f"{self.mlflow_tag_prefix}._key"] = key
        tags[f"{self.mlflow_tag_prefix}._timestamp"] = str(timestamp)

        # set optional tags
        tags[f"{self.mlflow_tag_prefix}._item_source"] = f"{socket.gethostname()}___{os.getpid()}"
        tags[mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME] = \
            f"{self.mlflow_tag_dict_name}{self.mlflow_tag_name_separator}{key}"
        tags[mlflow.utils.mlflow_tags.MLFLOW_USER] = f"{self.mlflow_tag_user}"
        if self.mlflow_tag_defaults is not None:
            for tag, value in self.mlflow_tag_defaults:
                tags[tag] = value
        if self.mlflow_tag_defaults is not None:
            for tag, value in self.mlflow_tag_defaults:
                tags[tag] = value

        # delete existing value / run before setting it, if update is not requested
        if not isinstance(value, MetaValue) or not value.update:
            for run in self._get_runs_with_name(key):
                # catching possibility that another process has already deleted the run
                # after we retrieved the corresponding runs
                try:
                    self.client.delete_run(run.info.run_id)
                except mlflow.exceptions.RestException as e:
                    if "Current state is deleted." in e.message:
                        warnings.warn("Run was already deleted: {run.info.run_id}")
                    else:
                        raise e

        # get run; the run exists if
        # * this happens if MetaValue.update=True
        # * OR when some other process created that run in the mean time
        run: mlflow.entities.Run = self.get_run(key)
        # TODO: race condition: what if someone deletes the run before we can finish with it?
        if run is None:
            # create run if the run is not there
            run = self.client.create_run(self.experiment.experiment_id, tags=tags)

        # noinspection PyBroadException
        try:

            # set meta data if applicable
            def set_run(func, values):
                if values is not None:
                    if not isinstance(values, list):
                        values = [values]
                    for args in values:
                        if isinstance(args, dict):
                            func(run.info.run_id, **args)
                        else:
                            if not isinstance(args, tuple):
                                args = (args, )
                            func(run.info.run_id, *args)

            if isinstance(value, MetaValue):
                if value.update:
                    set_run(self.client.set_tag, list(tags.items()))

                set_run(self.client.set_tag, value.tags)
                set_run(self.client.log_param, value.params)
                set_run(self.client.log_metric, value.metrics)
                set_run(self.client.log_artifact, value.artifacts)
                set_run(self.client.set_terminated, value.status)
            else:
                # set finished by default
                self.client.set_terminated(run.info.run_id, "FINISHED")

            # custom logging
            value = value.value if isinstance(value, MetaValue) else value
            with mlflowhelper.tracking.artifactmanager.ArtifactManager(self.client) as afm:
                self._custom_logging(self.mlflow_custom_logger, afm, run, key, value)
                if self.mlflow_pickler is not None:
                    with afm.managed_artifact("value.pickle", dst_run_id=run.info.run_id) as a:
                        self.mlflow_pickler.dump(value, a.get_path())

            return value, timestamp

        except Exception as e:
            # clean up run in case of error
            try:
                self.client.delete_run(run.info.run_id)
            except Exception as e2:
                warnings.warn(f"Clean-up failed: {e2}")
            raise e

    def _custom_logging(self, logger, artifact_manager, run, key, value):
        if logger is not None:
            if callable(logger):
                logger(self.client, artifact_manager, run, key, value)
            else:
                logger.log(self.client, artifact_manager, run, key, value)

    def _del_runs_with_name(self, name):
        for run in self._get_runs_with_name(name):
            self.client.delete_run(run.info.run_id)

    def _get_runs_with_name(self, name):
        select_only_finished_runs = "AND attributes.status = 'FINISHED'" if self.only_load_finished_runs else ""
        return self.client.search_runs(
            self.experiment.experiment_id,
            filter_string=f"tags.`{self.mlflow_tag_prefix}._class` = '{DICT_IDENTIFIER}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._name`='{self.mlflow_tag_dict_name}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._key`='{name}' "
                          f"{select_only_finished_runs}",
            max_results=SEARCH_MAX_RESULTS_THRESHOLD)

    def _get_all_runs(self):
        select_only_finished_runs = "AND attributes.status = 'FINISHED'" if self.only_load_finished_runs else ""
        return self.client.search_runs(
            self.experiment.experiment_id,
            filter_string=f"tags.`{self.mlflow_tag_prefix}._class` = '{DICT_IDENTIFIER}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._name`='{self.mlflow_tag_dict_name}' "
                          f"{select_only_finished_runs}",
            max_results=SEARCH_MAX_RESULTS_THRESHOLD)

    def _load_artifact(self, run_id):
        with mlflowhelper.tracking.artifactmanager.ArtifactManager(self.client) as afm:
            with afm.managed_artifact("value.pickle", src_run_id=run_id, skip_log=True) as a:
                value = self.mlflow_pickler.load(a.get_path())
                return value

    def _get_items(self, return_runs=False):
        """Loads all items directly from MlFlow skipping any local cache."""
        for r in self._get_all_runs():
            key = r.data.tags[f"{self.mlflow_tag_prefix}._key"]
            value = self._load_artifact(r.info.run_id)
            if return_runs:
                yield key, value, r
            else:
                yield key, value

    def search(self, regexp=None, tags=None):
        """Searches for values. NOT IMPLEMENTED YET"""
        # TODO: implement search
        raise NotImplementedError("Not implemented yet")

    def apply_logging(self, logging):
        """Applies logging to elements. This allows to add additional metrics, parameters, tags, etc. to values
        even after they have been set."""
        for key, value, run in self._get_items(return_runs=True):
            with mlflowhelper.tracking.artifactmanager.ArtifactManager(self.client) as afm:
                self._custom_logging(logging, afm, run, key, value)

    def get_run(self, k):
        runs = self._get_runs_with_name(k)
        if len(runs) == 0:
            return None
        elif len(runs) == 1:
            return runs[0]
        else:
            raise KeyError(f"Too many entries for key `{k}`: {len(runs)}")

    def update_keys(self):
        """Fetches keys from MlFlow. Call this when you suspect that the dict might have been updated from elsewhere."""
        self.local_all_keys = set(v.data.tags[f"{self.mlflow_tag_prefix}._key"] for v in self._get_all_runs())

        # clean up mising values
        if self.local_cache:
            for k in list(self.local_values.keys()):
                if k not in self.local_all_keys:
                    del self.local_values[k]

        return self.local_all_keys

    def reset_cache(self):
        """
        Empty/reset cache.
        """
        self.local_values = dict()

    def fill_cache(self, reset_cache=False, drop_none=False, mode="silent"):
        """
        Convenience method to update cache.
        Note that existing values will not be updated unless the whole cache is reset (`reset_cache`).
        """

        if mode not in ["tqdm", "print", "silent"]:
            raise ValueError(f"Invalid mode: {mode}")

        if not self.local_cache:
            raise ValueError("Local cache is not enabled.")
        else:
            if reset_cache:
                self.reset_cache()

            sync_mode_backup = self.sync_mode
            self.sync_mode = None
            try:
                self.update_keys()
                if mode == "tqdm":
                    import tqdm
                    for k in tqdm.tqdm(list(self.keys())):
                        v = self[k]
                        if drop_none and v is None:
                            warnings.warn(f"Dropping `None` entry: {k}")
                            del self[k]
                elif mode == "print" or mode == "silent":
                    for k in list(self.keys()):
                        if mode == "print":
                            print("Loading:", k)
                        v = self[k]
                        if drop_none and v is None:
                            warnings.warn(f"Dropping `None` entry: {k}")
                            del self[k]
            finally:
                self.sync_mode = sync_mode_backup

    def runs(self, include_values=False):
        """
        :param include_values:
        :return:
            * include_values=True  -> yield key, value, run
            * include_values=False -> yield key, run
        """
        if include_values:
            yield from self._get_items(return_runs=True)
        else:
            yield from self._get_all_runs()

    def delete_all(self, verbose=False):
        if verbose:
            import tqdm
            it = tqdm.tqdm(list(self.keys()))
        else:
            it = list(self.keys())

        for k in it:
            del self[k]

    # @contextmanager
    # def _acquire_run(self, key, create=False, timeout=3000, delay=200):
    #     """Attempt to avoid inconsistent states caused by race conditions. Probably still doesn't solve all of them:
    #         lock1 check
    #                     lock2 check
    #         lock1 start >
    #                     lock2 start >                   # let this be slow for some reason
    #                                     > lock1 set
    #         lock1 read >
    #                                     < lock1 send
    #         lock1 acquired <
    #                                     > lock2 set     # this is the issue: MlFlow server does not recognize locks ;)
    #                     lock2 read >
    #                                     < lock2 send
    #                     lock2 acquired <
    #
    #         * lock2 should have failed
    #         * if can happens if read and write are asynchronous on server
    #         TODO: Better ideas without an external lock service?
    #           YES! We can basically always only return the most recent entry for each key.
    #           That way we never have concurrent access on each key and even have a kind of version control.
    #           The old keys then need to be purged every now and then to keep the queries efficient (
    #           check indices in database to make sure that we are doing the most efficient thing).
    #     """
    #
    #     start = time.time()
    #
    #     lock_tag = f"{self.mlflow_tag_prefix}._lock"
    #     lock_value = self.lock_uuid
    #
    #     run: typing.Union[str, mlflow.entities.Run, None] = "undefined"
    #     while run == "undefined":
    #
    #         # check timeout
    #         if time.time() - start > timeout / 1000:
    #             raise RuntimeError(
    #                 f"Acquiring run with name '{key}' took too long (> {timeout}). " \
    #                 f"Possible deadlock while acquiring run.")
    #
    #         # get run and acquire lock
    #         existing_runs = sorted(self._get_runs_with_name(key), key=lambda r: r.info.run_id)
    #         if len(existing_runs) == 0:
    #             if create:
    #
    #                 created_run = self.client.create_run(
    #                     self.experiment.experiment_id,
    #                     tags={lock_tag: lock_value})
    #
    #                 updated_runs = sorted(self._get_runs_with_name(key), key=lambda r: r.info.run_id)
    #
    #                 if lock_tag in updated_runs[0].data.tags and updated_runs[0].data.tags[lock_tag] == lock_value:
    #                     run = updated_runs[0]
    #                 else:
    #                     self.client.delete_run(created_run.info.run_id)
    #             else:
    #                 run = None
    #
    #         elif len(existing_runs) > 0:
    #
    #             if lock_tag not in existing_runs[0].data.tags:
    #
    #                 # try to lock run
    #                 # noinspection PyBroadException
    #                 try:
    #                     self.client.set_tag(existing_runs[0].info.run_id, lock_tag, lock_value)
    #                 except Exception:
    #                     # run may have been deleted in the meantime
    #                     pass
    #
    #                 # check whether locking worked
    #                 updated_runs = sorted(self._get_runs_with_name(key), key=lambda r: r.info.run_id)
    #
    #                 if len(updated_runs) > 0 \
    #                         and lock_tag in updated_runs[0].data.tags \
    #                         and updated_runs[0].data.tags[lock_tag] == lock_value:
    #                     run = existing_runs[0]
    #
    #         else:
    #             raise RuntimeError(f"More than one run with key: {key}")
    #
    #         # check if lock is acquired
    #         if run is not None:
    #             run = self.client.get_run(run.info.run_id)
    #             if run.data.tags[lock_tag] != lock_value:
    #                 run = "undefined"
    #
    #         time.sleep(np.random.randint(delay))
    #
    #     yield run
    #
    #     if run is not None:
    #         self.client.delete_tag(run.info.run_id, "lock")

    def __setitem__(self, k, v) -> None:
        # TODO: race conditions (acquire lock on run)
        if self.read_only:
            raise RuntimeError("Dict is in 'read only' mode.")

        """Sets a value, optionally with additional tags."""
        if len(self.local_all_keys) >= SEARCH_MAX_RESULTS_THRESHOLD:
            raise MemoryError(
                f"Exceeding size limit for retrieval ({SEARCH_MAX_RESULTS_THRESHOLD}). "
                "This may result in missing values. "
                "This is a known limitation and will have to be fixed "
                "if it becomes an issue in practice.")

        value, timestamp = self._log_value(k, v)
        self.local_all_keys.add(k)
        if self.local_cache:
            self.local_values[k] = (value, timestamp)

    def __delitem__(self, k) -> None:
        # TODO: race conditions (acquire lock on run)
        if self.read_only:
            raise RuntimeError("Dict is in 'read only' mode.")

        self._del_runs_with_name(k)
        self.local_all_keys.remove(k)
        if self.local_cache:
            if k in self.local_values:
                del self.local_values[k]

    def __getitem__(self, k):
        # check value cache first depending on settings
        if k in self and self.local_cache and k in self.local_values:
            local_value, local_timestamp = self.local_values[k]

            if self.sync_mode == "full":

                # get remote value and timestamp
                run: mlflow.entities.Run = self.get_run(k)
                remote_timestamp = int(run.data.tags.get(f"{self.mlflow_tag_prefix}._timestamp", "0"))

                # choose value to return based on timestamp
                if remote_timestamp > local_timestamp:
                    remote_value = self._load_artifact(run.info.run_id)
                    self.local_values[k] = (remote_value, remote_timestamp)
                    return remote_value
                else:
                    return local_value
            else:
                return local_value
        else:
            # TODO: maybe limit get run to key that we know exists?
            # TODO: We could also cache run ids together with keys for less requests.
            run: mlflow.entities.Run = self.get_run(k)
            if run is None:
                if hasattr(self.__class__, "__missing__"):
                    # noinspection PyUnresolvedReferences
                    return self.__class__.__missing__(self, k)
                else:
                    raise KeyError(f"Key does not exist: {k}")
            else:
                value = self._load_artifact(run.info.run_id)
                timestamp = int(run.data.tags.get(f"{self.mlflow_tag_prefix}._timestamp", "0"))
                if self.local_cache:
                    self.local_values[k] = value, timestamp
                return value

    def __len__(self) -> int:
        if self.sync_mode is not None:
            self.update_keys()
        return len(self.local_all_keys)

    def __iter__(self) -> Iterator:
        if self.sync_mode is not None:
            self.update_keys()
        return iter(self.local_all_keys)

    # TODO: Modify __contains__ to work correctly when __missing__ is present
    def __contains__(self, key):
        if self.sync_mode is not None:
            self.update_keys()
        return key in self.local_all_keys

    # Now, add the methods in dicts but not in MutableMapping
    def __repr__(self):
        return repr(self.local_all_keys)

    # def items(self):
    #     return MlflowDict.ItemsView(self)

    # class ItemsView(typing.MappingView, typing.Set):
    #
    #     __slots__ = ()
    #
    #     @classmethod
    #     def _from_iterable(self, it):
    #         return set(it)
    #
    #     def __contains__(self, item):
    #         key, value = item
    #         try:
    #             v = self._mapping[key]
    #         except KeyError:
    #             return False
    #         else:
    #             return v is value or v == value
    #
    #     def __iter__(self):
    #         for key in self._mapping:
    #             yield (key, self._mapping[key])
