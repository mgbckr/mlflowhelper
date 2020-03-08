import collections.abc
from typing import _KT, _VT, Iterator, _T_co, _VT_co

import mlflow


class MlflowDict(collections.abc.MutableMapping):

    def __init__(
            self,
            mlfow_client: mlflow.tracking.client.MlflowClient,
            mlflow_experiment_name="default",
            mlflow_tag_context="default",
            mlflow_tag_custom=None,
            mlflow_tag_prefix="_mlflowdict",
            local_cache=True,
            lazy_cache=True):

        self.client = mlfow_client

        # tags
        self.mlflow_tag_context = mlflow_tag_context
        self.mlflow_tag_custom = mlflow_tag_custom
        self.mlflow_tag_prefix = mlflow_tag_prefix

        # caching
        self.local_cache = local_cache
        self.lazy_cache = lazy_cache
        self.local_all_keys = None
        self.local_data = None

        # create experiment if it does not exist
        exp = self.client.get_experiment_by_name(mlflow_experiment_name)
        if exp is None:
            self.client.create_experiment(mlflow_experiment_name)
        self.experiment = exp

    def load_all(self):
        pass

    def _log_experiment(self, name, experiment):

        for run in self._get_experiment(name):
            self.client.delete_run(run.info.run_id)
        run = self.client.create_run(self.experiment.experiment_id)
        # self.client.arti

        if self.local_cache:
            self.local_data[name] = experiment

    def _get_experiment(self, name):
        # return self.client.search_runs(
        #     self.experiment,
        #     filter_string=f"tags.category=`experiments` "
        #                   f"AND tags.context=`{self.context}` "
        #                   f"AND tags.experiment_name=`{name}`")
        pass

    def __setitem__(self, k: _KT, v: _VT) -> None:
        pass

    def __delitem__(self, v: _KT) -> None:
        pass

    def __getitem__(self, k: _KT) -> _VT_co:
        pass

    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[_T_co]:
        pass

    # Modify __contains__ to work correctly when __missing__ is present
    def __contains__(self, key):
        return key in self.data

    # Now, add the methods in dicts but not in MutableMapping
    def __repr__(self):

        return repr(self.data)

    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["data"] = self.__dict__["data"].copy()
        return inst

    def copy(self):
        if self.__class__ is MlflowDict:
            return MlflowDict(self.data.copy())
        import copy
        data = self.data
        try:
            self.data = {}
            c = copy.copy(self)
        finally:
            self.data = data
        c.update(self)
        return c
