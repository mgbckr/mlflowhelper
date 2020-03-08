import collections.abc
import pickle
from typing import Iterator

import mlflow
import mlflowhelper.tracking.artifactmanager

DICT_IDENTIFIER = "mlflow.tracking.collections.MlflowDict"


class MlflowDict(collections.abc.MutableMapping):
    """
    TODO: clean up and document
    """

    def __init__(
            self,
            mlfow_client: mlflow.tracking.client.MlflowClient = None,
            mlflow_experiment_name="dict_db",
            mlflow_tag_context="default",
            mlflow_tag_defaults=None,
            mlflow_tag_prefix="_mlflowdict",
            local_cache=True,
            lazy_cache=True):

        if mlfow_client is None:
            self.client = mlflow.tracking.MlflowClient()
        else:
            self.client = mlfow_client
        self.artifact_manager = mlflowhelper.tracking.artifactmanager.ArtifactManager(self.client)

        # tags
        self.mlflow_tag_context = mlflow_tag_context
        self.mlflow_tag_defaults = mlflow_tag_defaults
        self.mlflow_tag_prefix = mlflow_tag_prefix

        # caching
        self.local_cache = local_cache
        self.lazy_cache = lazy_cache
        self.local_all_keys = None
        self.local_data = dict()

        # create experiment if it does not exist
        exp = self.client.get_experiment_by_name(mlflow_experiment_name)
        if exp is None:
            self.client.create_experiment(mlflow_experiment_name)
            exp = self.client.get_experiment_by_name(mlflow_experiment_name)
        self.experiment = exp

        # init
        self._init_keys()
        if not lazy_cache:
            self._init_data()

    def _init_keys(self):
        if self.local_all_keys is None:
            self.local_all_keys = set(
                v.data.tags[f"{self.mlflow_tag_prefix}._key"] for v in self._get_all_runs())

    def _init_data(self):
        self.local_data = dict(self._get_all_data())

    def _log_artifact(self, key, value, tags=None):

        for run in self._get_runs(key):
            self.client.delete_run(run.info.run_id)

        run = self.client.create_run(self.experiment.experiment_id)

        with self.artifact_manager as afm:
            with afm.managed_artifact("value.pickle", dst_run_id=run.info.run_id) as a:
                with open(a.get_path(), 'wb') as f:
                    pickle.dump(value, f)

        # set required tags
        self.client.set_tag(
            run.info.run_id, f"{self.mlflow_tag_prefix}._class", DICT_IDENTIFIER)
        self.client.set_tag(run.info.run_id, f"{self.mlflow_tag_prefix}._context", self.mlflow_tag_context)
        self.client.set_tag(run.info.run_id, f"{self.mlflow_tag_prefix}._key", key)

        # set other tags
        if self.mlflow_tag_defaults is not None:
            for tag, value in self.mlflow_tag_defaults:
                self.client.set_tag(run.info.run_id, tag, value)
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                self.client.set_tag(run.info.run_id, tag, value)

    def _del_artifacts(self, name):
        for run in self._get_runs(name):
            self.client.delete_run(run.info.run_id)

    def _get_runs(self, name):
        return self.client.search_runs(
            self.experiment.experiment_id,
            filter_string=f"tags.`{self.mlflow_tag_prefix}._class` = '{DICT_IDENTIFIER}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._context`='{self.mlflow_tag_context}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._key`='{name}'")

    def _get_all_runs(self):
        return self.client.search_runs(
            self.experiment.experiment_id,
            filter_string=f"tags.`{self.mlflow_tag_prefix}._class` = '{DICT_IDENTIFIER}' "
                          f"AND tags.`{self.mlflow_tag_prefix}._context`='{self.mlflow_tag_context}'")

    def _load_artifact(self, run_id):
        with self.artifact_manager as afm:
            with afm.managed_artifact("value.pickle", src_run_id=run_id, skip_log=True) as a:
                with open(a.get_path(), "rb") as f:
                    value = pickle.load(f)
                    return value

    def _get_all_data(self):
        for r in self._get_all_runs():
            key = r.data.tags[f"{self.mlflow_tag_prefix}._key"]
            value = self._load_artifact(r.info.run_id)
            yield key, value

    def search(self, regexp=None, tags=None):
        # TODO: implement search
        raise NotImplementedError("Not implemented yet")

    def set(self, k, v, tags=None):
        self._log_artifact(k, v, tags=tags)
        self.local_all_keys.add(k)
        if self.local_cache:
            self.local_data[k] = v

    def __setitem__(self, k, v) -> None:
        self.set(k, v, tags=None)

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
