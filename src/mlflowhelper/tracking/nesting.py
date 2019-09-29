import warnings


def flatten_tree_depth_first(tree, exclude=None, exclude_children=False, return_only_run_id=False, return_trace=False):
    """
    Flatten a run tree in a depth-first manner. May return a list of run ids or runs.

    :param tree: dict
        The tree to flatten
    :param exclude:
        A function to exclude nodes
    :param exclude_children: bool
        Whether to exclude children of excluded nodes
    :param return_only_run_id: bool
        Whether to return only ids
    :param return_trace: bool
        Whether to return a run id trace for each listed run
    :return: list
        list of runs or run ids or tuples (run id trace, run_id|run)
    """

    if exclude is None:
        def exclude(run):
            return False

    run_list = []

    def traverse(trace, parent_id, parent_wrapper):

        if not (exclude_children and exclude(parent_wrapper["run"])):
            for child_id, child_wrapper in parent_wrapper["children"].items():
                traverse(trace + [parent_id], child_id, child_wrapper)

        if not exclude(parent_wrapper["run"]):
            entity = parent_id if return_only_run_id else parent_wrapper["run"]
            if return_trace:
                run_list.append((trace + [parent_id], entity))
            else:
                run_list.append(entity)

    for run_id, run_wrapper in tree.items():
        traverse([], run_id, run_wrapper)

    return run_list


def build_tree(runs, only_finished=True, return_all_nodes=False, return_new_nodes=False, parent_tag="parent"):
    """
    Builds a tree based on the given runs.
    The parent/child structure is based on a customizable `parent` tag which contains the parent's `run_id`.

    :param runs:
        The runs to build the tree from
    :param only_finished:
        Whether to only keep FINISHED runs
    :param return_all_nodes:
        Whether to return a flat `dict` with all runs
    :param return_new_nodes:
        Whether to return the new runs added to optionally given data structures
    :param parent_tag: str
        The tag to base the parent/child relation on
    :return: dict or list[dict]
        tree | tree(,all_nodes)(,new_nodes)
    """

    root_nodes, all_nodes, new_nodes = _build_tree_add_runs_to_data_structures(
        runs,
        only_finished=only_finished,
        parent_tag=parent_tag)

    return_vars = [root_nodes]
    if return_all_nodes:
        return_vars.append(all_nodes)
    if return_new_nodes:
        return_vars.append(new_nodes)

    if return_all_nodes or return_new_nodes:
        return return_vars
    else:
        return root_nodes


def _build_tree_add_runs_to_data_structures(
        runs,
        root_nodes=None,
        all_nodes=None,
        new_nodes=None,
        only_finished=True,
        parent_tag="parent"):

    if root_nodes is None:
        root_nodes = {}
    if all_nodes is None:
        all_nodes = {}
    if new_nodes is None:
        new_nodes = {}

    if not isinstance(runs, list):
        runs = [runs]

    for run in runs:
        run_id = run.info.run_id
        if run_id not in all_nodes:
            if not only_finished or run.info.status == "FINISHED":
                run_wrapper = {
                    "run": run,
                    "children": {}
                }
                all_nodes[run_id] = run_wrapper
                root_nodes[run_id] = run_wrapper
                new_nodes[run_id] = run_wrapper

    for run_id in [r for r in root_nodes.keys()]:
        run_wrapper = root_nodes[run_id]
        tags = run_wrapper["run"].data.tags
        if parent_tag in tags:
            parent_id = tags[parent_tag]
            if parent_id in all_nodes:
                all_nodes[parent_id]["children"][run_id] = run_wrapper
                del root_nodes[run_id]

    return root_nodes, all_nodes, new_nodes


def get_children_recursively(
        client, runs, experiment_ids=None, filter_string=None, only_finished=True, parent_tag="parent", verbose=0):
    """
    Loads all children runs of the given set of runs and puts them in a tree.
    The parent/child structure is based on a customizable `parent` tag which contains the parent's `run_id`.

    :param client: mlflow.tracking.MlflowClient
        The MlflowClient to use
    :param runs:
        Runs to collect parents for
    :param experiment_ids: str or list[str]
        The ids of the experiments to search for children in
    :param filter_string: str
        Filter string used to search for children
    :param only_finished:
        Whether to process only finished runs
    :param parent_tag: str
        The parent tag to use for the parent/child relation
    :param verbose:
        `verbose=1` prints a detailed progress report
    :return:
        Either a list of runs (`recursive=False`) or a dict with the parent-tree structure
    """

    if not isinstance(runs, list):
        runs = [runs]

    new_nodes = {}
    all_nodes = {}
    root_nodes = {}

    _build_tree_add_runs_to_data_structures(
        runs, root_nodes, all_nodes, new_nodes, only_finished=only_finished)

    if experiment_ids is None:
        experiment_ids = set([w["run"].info.experiment_id for w in all_nodes.values()])

    level = 0
    while len(new_nodes) > 0:
        if verbose > 0:
            print(f"Processing level {level}: {len(new_nodes)} nodes")

        parent_nodes = new_nodes
        new_nodes = {}
        for run_id, run_wrapper in parent_nodes.items():

            if verbose > 0:
                print(f"* querying children for {run_id} ... ", end="")

            children_filter_string = f"tags.{parent_tag}='{run_id}'"
            if filter_string is not None:
                children_filter_string += f" AND {filter_string}"

            children = client.search_runs(experiment_ids, filter_string=children_filter_string)
            if verbose > 0:
                print(f"found: {len(children)}")

            _build_tree_add_runs_to_data_structures(
                children, root_nodes, all_nodes, new_nodes, only_finished=only_finished)

        level += 1

    return root_nodes


def get_parents_recursively(client, runs, only_finished=True, parent_tag="parent", verbose=0):
    """
    Loads all parent runs of the given set of runs and puts them in a tree.
    The parent structure is based on a customizable `parent` tag which contains the parent's `run_id`.

    :param client: mlflow.tracking.MlflowClient
        The MlflowClient to use
    :param runs:
        Runs to collect parents for
    :param only_finished:
        Whether to process only finished runs
    :param parent_tag: str
        The parent tag to use for the parent/child relation
    :param verbose:
        `verbose=1` prints a detailed progress report
    :return:
        Either a list of runs (`recursive=False`) or a dict with the parent-tree structure
    """

    if not isinstance(runs, list):
        runs = [runs]

    new_nodes = {}
    all_nodes = {}
    root_nodes = {}

    _build_tree_add_runs_to_data_structures(
        runs, root_nodes, all_nodes, new_nodes, only_finished=only_finished)

    level = 0
    while len(new_nodes) > 0:
        if verbose > 0:
            print(f"Processing level {level}: {len(new_nodes)} nodes")

        child_nodes = new_nodes
        new_nodes = {}
        parents = []

        if verbose > 0:
            print(f"* collecting parents for child nodes: {len(child_nodes)}")

        for run_id, run_wrapper in child_nodes.items():

            tags = run_wrapper["run"].data.tags
            if "parent" in tags:
                parent = client.get_run(tags[parent_tag])
                parents.append(parent)

        if verbose > 0:
            print(f"* found parents: {len(parents)}")

        _build_tree_add_runs_to_data_structures(
            parents, root_nodes, all_nodes, new_nodes, only_finished=only_finished)

        level -= 1

    return root_nodes


def delete_run_recursively(client, run_id, experiment_ids=None, dry_run=True, verbose=0):
    """
    Deletes a run and all it's children. For this function, the parent/children relationship is defined
    by the tag `parent` in child runs. The value of the `parent` tag is the `run_id` of the parent.

    :param client: mlflow.tracking.MlflowClient
        The MlflowClient to use
    :param run_id: str
        The run id to delete
    :param experiment_ids: list[str]
        Ids of experiment in which to search for runs
    :param dry_run:
        Whether to do a dry run (instead of deleting)
    :param verbose:
        `verbose=1` prints a detailed progress report
    :return:
        A tuple with ids: (collected ids, "successfully" deleted ids)
    """

    if dry_run:
        warnings.warn("Dry run: nothing is deleted! Set `dry_run=False` to actually delete things.")

    tree = get_children_recursively(
        client,
        client.get_run(run_id),
        experiment_ids=experiment_ids,
        verbose=verbose)
    ids_to_delete = flatten_tree_depth_first(tree, return_only_run_id=True)

    ids_deleted = []
    if not dry_run:

        if verbose > 0:
            print("Deleting ids:")

        for id_to_delete in ids_to_delete:
            if verbose > 0:
                print("*", id_to_delete)

            run_to_delete = client.get_run(id_to_delete)
            if run_to_delete.info.lifecycle_stage == "active":
                client.delete_run(id_to_delete)
                ids_deleted.append(id_to_delete)
            else:
                if verbose > 0:
                    print("  Run id '{}' skipped since it is not 'active' (lifecycle_stage={}).".format(
                        id_to_delete, run_to_delete.info.lifecycle_stage))

    return ids_to_delete, ids_deleted
