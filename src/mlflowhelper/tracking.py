import inspect
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Optional

import mlflow
from mlflow.entities.run_status import RunStatus

_artifact_manager = None  # type: Optional[ArtifactManager]


def set_tracking_uri(tracking_uri):
    """Sets the tracking URI analogously to `mlflow.set_tracking_uri` but supports special keywords.

    Parameters
    ----------
    tracking_uri: str
        "file"=None, "localhost"="http://localhost:5000"

    Returns
    -------
    None
    """
    mlflow.set_tracking_uri(_tracking_uri(tracking_uri))


def get_or_create_experiment(experiment_name, tracking_uri=None):
    """Get the experiment id of the given experiment name. If it does not exist, it is created.

    Parameters
    ----------
    experiment_name: str
        Experiment name to get the id for

    tracking_uri: str, optional, default: None
        See `.set_tracking_uri`.

    Returns
    -------
    int
        Experiment id for the given experiment name

    """
    tracking_uri = _tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    if experiment_id is None:
        experiment_id = client.create_experiment(experiment_name)

    return experiment_id


def set_experiment(experiment_name):
    """Same as `mlflow.set_experiment` but also returns the experiment id.
    Like `mlflow.set_experiment` it also creates the experiment if it does not exist.

    Parameters
    ----------
    experiment_name: str
        Experiment name

    Returns
    -------
    int
        The experiment id

    """
    mlflow.set_experiment(experiment_name)
    return get_or_create_experiment(experiment_name)


def start_run(run_id=None, experiment_id=None, run_name=None, nested=False, tmp_dir=None, delete_tmp_dir=True):
    """
    Same as ``mlflow.start_run`` but also initializes an artifact manager that
    1) manages a temporary directory for artifacts,
    2) simplifies logging artifacts, and
    2) allows do load artifacts from previous runs instead of recalculating them.

    Parameters
    ----------
    run_id: str, optional, default: None
        See ``mlflow.start_run``
    experiment_id: str, optional, default: None
        See ``mlflow.start_run``
    run_name: str, optional, default: None
        See ``mlflow.start_run``
    nested: bool, optional, default: False
        See ``mlflow.start_run``
    tmp_dir: str, optional: default: None
        The temporary directory for this run. If left unset, a unique, random temporary directory will be created.
    delete_tmp_dir: bool, optional, default: True
        Whether to delete the temporary directory after the run is finished.

    Returns
    -------
    ActiveRunWrapper

    See Also
    --------
    mlflow.start_run
    mlflowhelper.managed_artifact
    mlflowhelper.managed_artifacts

    Examples
    --------
    >>> import nalabtools.utils.mlflowhelper as mlflowhelper
    >>> from matplotlib import pyplot as plt
    >>> with mlflowhelper.start_run():
    >>>     with mlflowhelper.managed_artifact("plot.png") as artifact:
    >>>         fig = plt.figure()
    >>>         plt.scatter([1,2,3], [1,2,3])
    >>>         fig.savefig(artifact.get_path())
    """

    global _artifact_manager

    active_run = mlflow.start_run(run_id, experiment_id, run_name, nested)

    artifact_manager = None
    if _artifact_manager is None:
        artifact_manager = ArtifactManager(tmp_dir=tmp_dir, delete_tmp_dir=delete_tmp_dir)
        _artifact_manager = artifact_manager

    return ActiveRunWrapper(active_run, artifact_manager)


def end_run(status=RunStatus.to_string(RunStatus.FINISHED)):
    """Same calls `mlflow.end_run`; nothing else.

    Parameters
    ----------
    status: mlflow.RunStatus

    Returns
    -------
    None
    """
    mlflow.end_run(status=status)


def set_load(run_id, stages='all'):
    """Set stages that should be loaded from previous run.

    Parameters
    ----------
    run_id: str
        Run id to lead from
    stages: str or list[str], optional, default: "all"
        Stages to load

    Returns
    -------
    None
    """
    if _artifact_manager is None:
        raise Exception("`mlflowhelper.set_load` can only be called within a run started my `mlflowhelper.start_run`")
    _artifact_manager.set_load(run_id, stages)


def get_loading_information():
    """Get logging information of active run. TODO: Should probably moved to `ArtifactManager`"""
    if _artifact_manager is None:
        raise Exception("""`mlflowhelper.get_loading_information` can only be
            called within a run started my `mlflowhelper.start_run`""")
    return {
        "experiment_id": mlflow.active_run().info.experiment_id,
        "run_id": _artifact_manager.run_id
    }


def get_artifact_manager():
    """Get artifact manager for active run."""
    return _artifact_manager


def set_skip_log(stages='all'):
    """Set stages for which logging should be skipped.

    Parameters
    ----------
    stages: str or list[str], optional, default: 'all'
    """
    if _artifact_manager is None:
        raise Exception(
            "`mlflowhelper.set_skip_log` can only be called within a run started my `mlflowhelper.start_run`")
    _artifact_manager.set_skip_log(stages)


def managed_artifact(
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
        local path of the artifact relative to the path used by the responsible ArtifactManager
    artifact_path: str
    stage: str, optional, default: None
        name of the stage used to configure loading and logging behavior
    load: bool, optional, default: None
        whether the artifact should be loaded; If not `None` this overwrites settings in the ArtifactManager.
    skip_log: bool, optional, default: None
        whether the artifact should be logged; If not `None` this overwrites settings in the ArtifactManager.
    delete: bool, optional, default: None
        whether the artifact should be deleted after it was logged;
        If not `None` this overwrites settings in the ArtifactManager.


    Returns
    -------
    ManagedResource

    See Also
    --------
    .managed_artifact_dir
    .ArtifactManager.managed_artifact
    .ArtifactManager.managed_artifact_dir


    Examples
    --------
    Simply logging several artifacts in a directory:
    >>> import nalabtools.utils.mlflowhelper as mlflowhelper
    >>> import pandas as pd
    >>> from matplotlib import pyplot as plt
    >>> data = pd.read_csv("data.csv")
    >>>
    >>> with mlflowhelper.start_run():
    >>>     with mlflowhelper.managed_artifact("plots") as artifact:
    >>>         fig = plt.figure()
    >>>         plt.plot(data["x"], data["y1"])
    >>>         fig.savefig(artifact.get_path())

    Loading data from previous run:
    >>> import nalabtools.utils.mlflowhelper as mlflowhelper
    >>> import pandas as pd
    >>>
    >>> with mlflowhelper.start_run():
    >>>     # activate loading from previous run
    >>>     mlflowhelper.set_load(run_id="e1363f760b1e4ab3a9e93f856f2e9341", stages=["load_data"])
    >>>     with mlflowhelper.managed_artifact("data.csv", stage="load_data") as artifact:
    >>>         if artifact.loaded:
    >>>             # load artifact
    >>>             data = pd.read_csv(artifact.get_path())
    >>>         else:
    >>>             # create and save artifact
    >>>             data = pd.read_csv("/shared/dir/data.csv").sample(frac=1)
    >>>             data.to_csv(artifact.get_path())
    """
    if _artifact_manager is None:
        raise Exception(
            "`mlflowhelper.managed_artifact` can only be called within a run started my `mlflowhelper.start_run`")
    return _artifact_manager.managed_artifact(
            file_path,
            artifact_path=artifact_path,
            stage=stage,
            load=load,
            skip_log=skip_log,
            delete=delete)


def managed_artifact_dir(
        dir_path,
        stage=None,
        load=None,
        skip_log=None,
        delete=None):
    """

    Parameters
    ----------
    dir_path: str
        local path of the artifact directory relative to the path used by the responsible ArtifactManager
    stage: str, optional, default: None
        name of the stage used to configure loading and logging behavior
    load: bool, optional, default: None
        whether the artifact directory should be loaded; If not `None` this overwrites settings in the ArtifactManager.
    skip_log: bool, optional, default: None
        whether the artifact directory should be logged; If not `None` this overwrites settings in the ArtifactManager.
    delete: bool, optional, default: None
        whether the artifact directory should be deleted after it was logged;
        If not `None` this overwrites settings in the ArtifactManager.

    Returns
    -------
    ManagedResource

    See Also
    --------
    .managed_artifact
    .ArtifactManager.managed_artifact
    .ArtifactManager.managed_artifact_dir

    Examples
    --------
    Simply logging several artifacts in a directory:
    >>> import nalabtools.utils.mlflowhelper as mlflowhelper
    >>> import pandas as pd
    >>> from matplotlib import pyplot as plt
    >>> data = pd.read_csv("data.csv")
    >>>
    >>> with mlflowhelper.start_run():
    >>>     with mlflowhelper.managed_artifact_dir("plots") as artifact_dir:
    >>>         fig = plt.figure()
    >>>         plt.plot(data["x"], data["y1"])
    >>>         fig.savefig(artifact_dir.get_path("fig1.png"))
    >>>         fig = plt.figure()
    >>>         plt.plot(data["x"], data["y2"])
    >>>         fig.savefig(artifact_dir.get_path("fig2.png"))

    Loading data from previous run:
    >>> import nalabtools.utils.mlflowhelper as mlflowhelper
    >>> import pandas as pd
    >>>
    >>> with mlflowhelper.start_run():
    >>>     # activate loading from previous run
    >>>     mlflowhelper.set_load(run_id="e1363f760b1e4ab3a9e93f856f2e9341", stages=["load_data"])
    >>>     with mlflowhelper.managed_artifact_dir("data", stage="load_data") as artifact_dir:
    >>>         train_path = artifact_dir.get_path("test.csv")
    >>>         test_path = artifact_dir.get_path("train.csv")
    >>>         if artifact_dir.loaded:
    >>>             # load artifacts
    >>>             train = pd.read_csv(train_path)
    >>>             test = pd.read_csv(test_path)
    >>>         else:
    >>>             data = pd.read_csv("/shared/dir/data.csv").sample(frac=1)
    >>>             train = data.iloc[:100,:]
    >>>             test = data.iloc[100:,:]
    >>>             # save artifacts
    >>>             train.to_csv(train_path)
    >>>             test.to_csv(test_path)
    """
    if _artifact_manager is None:
        raise Exception(
            "`mlflowhelper.managed_artifacts` can only be called within a run started my `mlflowhelper.start_run`")
    return _artifact_manager.managed_artifact_dir(
            dir_path,
            stage=stage,
            load=load,
            skip_log=skip_log,
            delete=delete)


def log_vars(
        include=None, exclude=None,
        include_args=True, include_varargs=True, include_kwargs=True, include_locals=False,
        varargs_prefix="vararg_",
        verbose=0):
    """Log variables in the local scope.

    Parameters
    ----------
    include: list[str], optional, default: None
        List of variables to include; Only one, `include` OR `exclude`, can be specified.
    exclude: list[str], optional, default: None
        List of variables to exclude; Only one, `include` OR `exclude`, can be specified.
    include_args: bool
        Whether to include regular arguments of the enclosing function
    include_varargs: bool
        Whether to include varargs of the enclosing function
    include_kwargs: bool
        Whether to include keyword arguments of the enclosing function
    include_locals: bool
        Whether to include local variables besides the function arguments of the enclosing function
    varargs_prefix: str, optional, default: "vararg_"
        Prefix for (unnamed) varargs
    verbose: int, optional, default: 0
        1 will print the logged arguments

    Returns
    -------
    dict
        Dict containing the logged arguments and their values

    """

    if include is not None and exclude is not None:
        raise Exception("`include` and `exclude` have been set. However, only one can we not `None`.")

    calling_frame = inspect.stack()[1].frame
    arg_info = inspect.getargvalues(calling_frame)

    args = {}

    if include_args:
        for arg in arg_info.args:
            args[arg] = arg_info.locals[arg]

    if include_kwargs and arg_info.keywords is not None:
        for k, v in arg_info.locals[arg_info.keywords].items():
            args[k] = v

    if include_varargs is not None and arg_info.varargs is not None:
        for i, v in enumerate(arg_info.locals[arg_info.varargs]):
            args["{}{}".format(varargs_prefix, i)] = v

    if include_locals:
        skip = arg_info.args
        if arg_info.varargs is not None:
            skip.append(arg_info.varargs)
        if arg_info.keywords is not None:
            skip.append(arg_info.keywords)

        for k, v in arg_info.locals.items():
            if k not in skip:
                args[k] = v

    if include is not None:
        exclude = []
        for k, v in args.items():
            if k not in include:
                exclude.append(k)

    if exclude is not None:
        for arg in exclude:
            del args[arg]

    if verbose:
        print("mlflowhelper: Logging variables:")
        for k, v in args.items():
            print("  * {}={}".format(k, v))

    mlflow.log_params(args)
    return args


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

    def set_load(self, run_id, stages='all'):
        self.run_id = run_id
        self.stages_load = stages

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


def _tracking_uri(tracking_uri):
    """Helper function for URI shorthands.

    Parameters
    ----------
    tracking_uri: str
        "file" will translate to `None`,
        "localhost" to "http://localhost:5000", and
        "localhost-2" to "http://localhost:5002".

    Returns
    -------
    str or None
    """
    if tracking_uri == "file":
        tracking_uri = None
    elif tracking_uri is not None and tracking_uri.startswith("localhost"):
        split = tracking_uri.split("-")
        port = 5000
        if len(split) > 1:
            port += int(split[1])
        tracking_uri = "http://localhost:{}".format(port)
    else:
        tracking_uri = tracking_uri
    return tracking_uri
