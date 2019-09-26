import os
import shutil
import tempfile
from contextlib import contextmanager

import mlflow


class ArtifactManager(object):

    def __init__(self, client=None, tmp_dir=None, delete_tmp_dir=True):
        """
        Parameters
        ----------
        client: mlflow.tracking.MlflowClient, optional, default: None
            If `client=None` the client is initialized via `mlflow.tracking.MlflowClient()`.
        tmp_dir: str, optional, default: None
            If `tmp_dir=None`, `ArtifactManager.init` will create a
            temporary directory via `tempfile.mkdtemp(prefix="mlflowhelper_")`
        delete_tmp_dir: bool, optional, default: True
            Whether to delete the temporary directory on
            cleanup (see `ArtifactManager.cleanup` or `ArtifactManager.__exit__`).
        """

        if client is None:
            self.client = mlflow.tracking.MlflowClient()
        else:
            self.client = client

        self.tmp_dir = tmp_dir
        self.delete_tmp_dir = delete_tmp_dir

        self.run_id = None
        self.stages_load = None
        self.stages_skip_log = None

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return exc_type is None

    def init(self):
        """Initializes and creates a temporary directory (implicitly) set via the constructor.

        Returns
        -------
        None

        """
        if self.tmp_dir is None:
            self.tmp_dir = tempfile.mkdtemp(prefix="mlflowhelper_")
        else:
            os.makedirs(self.tmp_dir, exist_ok=True)

    def cleanup(self):
        """ Delete temporary directory if `self.delete_tmp_dir` is set to `True`.

        Returns
        -------
        None

        """
        if self.delete_tmp_dir:
            shutil.rmtree(self.tmp_dir)

    def set_skip_log(self, stages='all'):
        self.stages_skip_log = stages
        return self

    def set_load(self, run_id, stages='all'):
        self.run_id = run_id
        self.stages_load = stages
        return self

    def will_load_stages(self, stages):

        if isinstance(stages, str):
            stages = [stages]

        if self.run_id is not None and self.stages_load is not None:
            return self.stages_load == 'all' or all([stage in self.stages_load for stage in stages])
        else:
            return False

    def _stage_load(self, stage, load=None):

        # no run id no loading artifacts
        if self.run_id is None:
            return False

        # if load is given it overwrites everything
        if load is not None:
            return load

        return (stage is not None
                and self.stages_load is not None
                and (self.stages_load == "all" or stage in self.stages_load))

    def _stage_skip_log(self, stage, skip_log=None):

        # if log is given it overwrites everything
        if skip_log is not None:
            return skip_log

        return (stage is not None
                and self.stages_skip_log is not None
                and (self.stages_skip_log == "all" or stage in self.stages_skip_log))

    @contextmanager
    def managed_artifact(
            self,
            file_path,
            artifact_path=None,
            stage=None,
            load=None,
            skip_log=None,
            delete=None):
        """

        Parameters
        ----------
        file_path: str
        artifact_path: str, optional, default: None
        stage: str, optional, default: None
        load: bool, optional, default: None
        skip_log: bool, optional, default: None
        delete: bool, optional, default: None

        Yields
        -------
        ManagedResource

        See Also
        --------
        .managed_artifact
        """

        load = self._stage_load(stage, load=load)
        skip_log = self._stage_skip_log(stage, skip_log=skip_log)
        if delete is None:
            delete = True

        tmp_file = os.path.join(self.tmp_dir, file_path)
        dir_name = os.path.dirname(tmp_file)
        os.makedirs(dir_name, exist_ok=True)

        # download artifact if applicable
        if load:

            base_name = os.path.basename(file_path)

            if artifact_path is None:
                download_path = base_name
            else:
                download_path = artifact_path
            self.client.download_artifacts(self.run_id, download_path, dir_name)

        try:
            yield ManagedResource(tmp_file, load, skip_log, is_dir=False, manager=self)
        finally:
            if not skip_log:
                mlflow.log_artifact(tmp_file, artifact_path=artifact_path)
            if delete:
                os.remove(tmp_file)

    @contextmanager
    def managed_artifact_dir(
            self,
            dir_path,
            stage=None,
            load=None,
            skip_log=None,
            delete=None):
        """

        Parameters
        ----------
        dir_path: str
        stage: str, optional, default: None
        load: bool, optional, default: None
        skip_log: bool, optional, default: None
        delete: bool, optional, default: None

        Yields
        -------
        ManagedResource

        See Also
        --------
        .managed_artifact_dir
        """

        while dir_path[-1] == "/":
            dir_path = dir_path[:-1]

        load = self._stage_load(stage, load=load)
        skip_log = self._stage_skip_log(stage, skip_log=skip_log)
        if delete is None:
            delete = True

        tmp_dir = os.path.join(self.tmp_dir, dir_path)
        os.makedirs(tmp_dir, exist_ok=True)

        # download artifact if applicable
        if load:
            # download from artifact_repo:dir_path to local:self.tmp_dir/dir_path
            # TODO: here, we worked around weird a behavior of download_artifacts where it it downloads
            # prefix/last_dir to dst_path/last_dir ... this may actually be due to a missing trailing slash?
            # I have to test this eventually ... for now we strip those strip slashes!
            self.client.download_artifacts(
                self.run_id,
                path=dir_path,
                dst_path=os.path.join(*os.path.split(tmp_dir)[:-1]))

        try:
            yield ManagedResource(tmp_dir, load, skip_log, is_dir=True, manager=self)
        finally:
            if not skip_log:
                # saves everything from local:self.tmp_dir/dir_path to artifact_repo:dir_path
                mlflow.log_artifacts(tmp_dir, artifact_path=dir_path)
            if delete:
                shutil.rmtree(tmp_dir)


class ManagedResource(object):

    def __init__(self, path, loaded, skip_log, is_dir, manager):
        """

        Parameters
        ----------
        path: str
            The path of the resource
        loaded: bool
            Whether the resource will be loaded
        skip_log: bool
            Whether the resource will belogged
        is_dir: bool
            Whether the resource is a directory
        manager: ArtifactManager
            The ArtifactManager responsible for the resource
        """
        self._path = path
        self.loaded = loaded
        self.skip_log = skip_log
        self.is_dir = is_dir
        self.manager = manager

    def get_path(self, file_path=None):
        """
        Get local path of the current resource.
        If the resource is a directory, a relative `file_path` (relative to the managed directory)
        can be specified to get an absolute path to the file.

        Parameters
        ----------
        file_path: str, optional, default: None

        Returns
        -------
        str
            The absolute path to the specified resource.
            If this managed resource is a file, then the file path is returned.
            If this managed resource is a directory and the `file_path` parameter is `None`.
            then the path to this dorectory is returned.
            And if this managed resource is a directory and the `file_path` parameter is not `None`,
            then the absolute path to that file is given, i.e., `<resource path>/<file_path>`.

        """

        if self.is_dir:
            if file_path is None:
                return self._path
            else:
                path = os.path.join(self._path, file_path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                return path
        else:
            if file_path is not None:
                raise Exception(
                    "`file_path` parameter must be `None` for a file resource. You specified: {}".format(file_path))
            else:
                return self._path


class ActiveRunWrapper:
    """Wraps an `mlflow.ActiveRun` to support artifact management via an `.ArtifactManager`.
    """

    def __init__(self, active_run, artifact_manager):
        self.active_run = active_run
        self.artifact_manager = artifact_manager

    def __enter__(self):
        self.active_run.__enter__()
        if self.artifact_manager is not None:
            self.artifact_manager.init()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.artifact_manager is not None:
            self.artifact_manager.cleanup()
        self.active_run.__exit__(exc_type, exc_val, exc_tb)
