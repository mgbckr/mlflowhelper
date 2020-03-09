import os
import shutil
import tempfile
from contextlib import contextmanager

import mlflow


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


class ArtifactManager(object):

    def __init__(
            self,
            client: mlflow.tracking.MlflowClient = None,
            tmp_dir=None,
            delete_tmp_dir=True,
            src_run_id_stages=None,
            stages_skip_log=None):
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
        src_run_id_stages: str, optional, default: None
            A source run id for each stage. Overwrites `src_run_id_default`.
        stages_skip_log: str or list of str, optional, default: None
            Controls whether data is logged for each stage. If set to `all` logging is skipped for all. If set to
            a list of strings (naming stages) only those stages are skipped.
        """

        if client is None:
            self.client = mlflow.tracking.MlflowClient()
        else:
            self.client = client

        self.tmp_dir = tmp_dir
        self.delete_tmp_dir = delete_tmp_dir

        self.src_run_id_stages = src_run_id_stages
        self.stages_skip_log = stages_skip_log

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
            self.tmp_dir = None

    def will_load_stages(self, stages):

        if isinstance(stages, str):
            stages = [stages]

        if self.src_run_id_stages is not None:
            return all([stage in self.src_run_id_stages and self.src_run_id_stages[stage] is not None
                        for stage in stages])
        else:
            return False

    def set_src_run_id(self, run_id, stages=None):
        if stages is None:
            for stage in self.src_run_id_stages.keys():
                self.src_run_id_stages[stage] = run_id
        elif isinstance(stages, str):
            self.src_run_id_stages[stages] = run_id
        else:
            for stage in stages:
                self.src_run_id_stages[stage] = run_id

    def _get_run_id_for_stage_loading(self, stage, src_run_id=None):

        # overwrites other run ids
        if src_run_id is not None:
            return src_run_id

        elif self.src_run_id_stages is not None and stage in self.src_run_id_stages:
            return self.src_run_id_stages[stage]

        else:
            return None

    def _skip_logging_for_stage(self, stage, skip_log=None):

        # if log is given it overwrites everything
        if skip_log is not None:
            return skip_log

        return (stage is not None
                and self.stages_skip_log is not None
                and (self.stages_skip_log == "all" or stage in self.stages_skip_log))

    @staticmethod
    def _get_dst_run_id(dst_run_id=None):
        if dst_run_id is not None:
            return dst_run_id
        else:
            active_run = mlflow.active_run()
            if active_run is None:
                raise Exception("No run id given and no active run found!")
            else:
                return active_run.info.run_id

    @contextmanager
    def managed_artifact(
            self,
            file_path,
            artifact_path=None,
            stage=None,
            src_run_id=None,
            dst_run_id=None,
            skip_log=None,
            delete=None) -> ManagedResource:
        """

        Parameters
        ----------
        file_path: str
        artifact_path: str, optional, default: None
        stage: str, optional, default: None
        src_run_id: str, optional, default: None
        dst_run_id: str, optional, default: None
        skip_log: bool, optional, default: None
        delete: bool, optional, default: None

        Yields
        -------
        ManagedResource

        See Also
        --------
        .managed_artifact
        """

        src_run_id = self._get_run_id_for_stage_loading(stage, src_run_id=src_run_id)
        skip_log = self._skip_logging_for_stage(stage, skip_log=skip_log)
        if delete is None:
            delete = True

        tmp_file = os.path.join(self.tmp_dir, file_path)
        dir_name = os.path.dirname(tmp_file)
        os.makedirs(dir_name, exist_ok=True)

        # download artifact if applicable
        if src_run_id is not None:

            base_name = os.path.basename(file_path)

            if artifact_path is None:
                download_path = base_name
            else:
                download_path = artifact_path
            self.client.download_artifacts(src_run_id, download_path, dir_name)

        try:
            yield ManagedResource(tmp_file, src_run_id is not None, skip_log, is_dir=False, manager=self)
        finally:
            if not skip_log:
                if not os.path.exists(tmp_file):
                    raise FileNotFoundError(f"Artifact not found (`{file_path}`). Did you forget to create it?")
                run_id = ArtifactManager._get_dst_run_id(dst_run_id=dst_run_id)
                self.client.log_artifact(run_id, tmp_file, artifact_path=artifact_path)
            if delete:
                os.remove(tmp_file)

    @contextmanager
    def managed_artifact_dir(
            self,
            dir_path,
            stage=None,
            src_run_id=None,
            dst_run_id=None,
            skip_log=None,
            delete=None) -> ManagedResource:
        """

        Parameters
        ----------
        dir_path: str
        stage: str, optional, default: None
        src_run_id: str, optional, default: None
        dst_run_id: str, optional, default: None
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

        src_run_id = self._get_run_id_for_stage_loading(stage, src_run_id=src_run_id)
        skip_log = self._skip_logging_for_stage(stage, skip_log=skip_log)
        if delete is None:
            delete = True

        tmp_dir = os.path.join(self.tmp_dir, dir_path)
        os.makedirs(tmp_dir, exist_ok=True)

        # download artifact if applicable
        if src_run_id is not None:
            # download from artifact_repo:dir_path to local:self.tmp_dir/dir_path
            # TODO: here, I worked around weird a behavior of `download_artifacts` where it downloads
            # prefix/last_dir to dst_path/last_dir ... this may actually be due to a missing trailing slash?
            # I have to test this eventually ... for now we strip those strip slashes!
            self.client.download_artifacts(
                src_run_id,
                path=dir_path,
                dst_path=os.path.join(*os.path.split(tmp_dir)[:-1]))

        try:
            yield ManagedResource(tmp_dir, src_run_id is not None, skip_log, is_dir=True, manager=self)
        finally:
            if not skip_log:
                # saves everything from local:self.tmp_dir/dir_path to artifact_repo:dir_path
                run_id = ArtifactManager._get_dst_run_id(dst_run_id=dst_run_id)
                self.client.log_artifacts(run_id, tmp_dir, artifact_path=dir_path)
            if delete:
                shutil.rmtree(tmp_dir)


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
