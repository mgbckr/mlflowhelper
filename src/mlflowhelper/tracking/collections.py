import collections.abc
import pickle
import typing
import warnings
from abc import abstractmethod
from typing import Iterator

import mlflow
import mlflow.store.tracking
import mlflow.utils.mlflow_tags
import mlflowhelper.tracking.artifactmanager

DICT_IDENTIFIER = "mlflow.tracking.collections.MlflowDict"


class MlflowDict(collections.abc.MutableMapping):
    """
    Dict like class that persists elements to an Mlflow tracking server.
    WARNING: You need to explicitly set values again when you change them since the local cache is not automatically
    synchronized.
    """

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
            mlflow_pickler: typing.Union[Pickler, str] = 'pickle',
            local_cache=True,
            lazy_cache=True):
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
            See `MlflowDict.Pickler` for the interface.
        :param local_cache:
        :param lazy_cache:
        """

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
        else:
            self.mlflow_pickler = mlflow_pickler

        # caching
        self.local_cache = local_cache
        self.lazy_cache = lazy_cache
        self.local_all_keys = set(
            v.data.tags[f"{self.mlflow_tag_prefix}._key"] for v in self._get_all_runs())
        if not lazy_cache:
            self._init_data()
        else:
            self.local_data = dict()

        # check whether dict exists (initialized )
        if len(self.local_all_keys) > 0:
            warnings.warn("Dict already exists")

    def _init_data(self):
        self.local_data = dict(self._get_all_data())

    def _log_artifact(self, key, value, tags=None):

        for run in self._get_runs(key):
            self.client.delete_run(run.info.run_id)

        run = self.client.create_run(self.experiment.experiment_id)

        # noinspection PyBroadException
        try:
            # set required tags
            self.client.set_tag(
                run.info.run_id, f"{self.mlflow_tag_prefix}._class", DICT_IDENTIFIER)
            self.client.set_tag(run.info.run_id, f"{self.mlflow_tag_prefix}._name", self.mlflow_tag_dict_name)
            self.client.set_tag(run.info.run_id, f"{self.mlflow_tag_prefix}._key", key)

            # set other tags
            self.client.set_tag(
                run.info.run_id,
                mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME,
                f"{self.mlflow_tag_dict_name}{self.mlflow_tag_name_separator}{key}")
            self.client.set_tag(
                run.info.run_id,
                mlflow.utils.mlflow_tags.MLFLOW_USER,
                f"{self.mlflow_tag_user}")
            if self.mlflow_tag_defaults is not None:
                for tag, value in self.mlflow_tag_defaults:
                    self.client.set_tag(run.info.run_id, tag, value)
            if self.mlflow_tag_defaults is not None:
                for tag, value in self.mlflow_tag_defaults:
                    self.client.set_tag(run.info.run_id, tag, value)
            if tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                for tag in tags:
                    self.client.set_tag(run.info.run_id, tag, value)

            # custom logging
            with mlflowhelper.tracking.artifactmanager.ArtifactManager(self.client) as afm:
                self._custom_logging(self.mlflow_custom_logger, afm, run, key, value)
                if self.mlflow_pickler is not None:
                    with afm.managed_artifact("value.pickle", dst_run_id=run.info.run_id) as a:
                        self.mlflow_pickler.dump(value, a.get_path())

        except Exception as e:
            # clean up run in case of error
            self.client.delete_run(run.info.run_id)
            raise e

    def _custom_logging(self, logger, artifact_manager, run, key, value):
        if logger is not None:
            if callable(logger):
                logger(self.client, artifact_manager, run, key, value)
            else:
                logger.log(self.client, artifact_manager, run, key, value)

    def _del_artifacts(self, name):
        for run in self._get_runs(name):
            self.client.delete_run(run.info.run_id)

    def _get_runs(self, name):
        return self.client.search_runs(
            self.experiment.experiment_id,
            filter_string=f"tags.`{self.mlflow_tag_prefix}._class` = '{DICT_IDENTIFIER}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._name`='{self.mlflow_tag_dict_name}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._key`='{name}'")

    def _get_all_runs(self):
        return self.client.search_runs(
            self.experiment.experiment_id,
            filter_string=f"tags.`{self.mlflow_tag_prefix}._class` = '{DICT_IDENTIFIER}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._name`='{self.mlflow_tag_dict_name}'")

    def _load_artifact(self, run_id):
        with mlflowhelper.tracking.artifactmanager.ArtifactManager(self.client) as afm:
            with afm.managed_artifact("value.pickle", src_run_id=run_id, skip_log=True) as a:
                value = self.mlflow_pickler.load(a.get_path())
                return value

    def _get_all_data(self, include_runs=False):
        for r in self._get_all_runs():
            key = r.data.tags[f"{self.mlflow_tag_prefix}._key"]
            value = self._load_artifact(r.info.run_id)
            if include_runs:
                yield key, value, r
            else:
                yield key, value

    def search(self, regexp=None, tags=None):
        """Searches for values. NOT IMPLEMENTED YET"""
        # TODO: implement search
        raise NotImplementedError("Not implemented yet")

    def set_value(self, k, v, tags=None):
        """Sets a value, optionally with additional tags."""
        self._log_artifact(k, v, tags=tags)
        self.local_all_keys.add(k)
        if self.local_cache:
            self.local_data[k] = v

    def apply_logging(self, logging):
        """Applies logging to elements. This allows to add additional metrics, parameters, tags, etc. to values
        even after they have been set."""
        for key, value, run in self._get_all_data(include_runs=True):
            with mlflowhelper.tracking.artifactmanager.ArtifactManager(self.client) as afm:
                self._custom_logging(logging, afm, run, key, value)

    def __setitem__(self, k, v) -> None:
        self.set_value(k, v, tags=None)

    def __delitem__(self, k) -> None:
        self._del_artifacts(k)
        self.local_all_keys.remove(k)
        if self.local_cache:
            del self.local_data[k]

    def __getitem__(self, k):
        if self.local_data is not None and k in self.local_data:
            return self.local_data[k]
        else:
            runs = self._get_runs(k)
            if len(runs) > 1:
                raise Exception(f"Too many entries for key: `{k}`")
            elif len(runs) == 0:
                if hasattr(self.__class__, "__missing__"):
                    # noinspection PyUnresolvedReferences
                    return self.__class__.__missing__(self, k)
            if len(runs) == 1:
                run_id = runs[0].info.run_id
                value = self._load_artifact(run_id)
                if self.local_cache:
                    self.local_data[k] = value
                return value
            else:
                raise KeyError(f"Key does not exist: {k}")

    def __len__(self) -> int:
        return len(self.local_all_keys)

    def __iter__(self) -> Iterator:
        return iter(self.local_all_keys)

    # Modify __contains__ to work correctly when __missing__ is present
    def __contains__(self, key):
        return key in self.local_all_keys

    # Now, add the methods in dicts but not in MutableMapping
    def __repr__(self):
        return repr(self.local_all_keys)
