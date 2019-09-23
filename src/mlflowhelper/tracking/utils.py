import warnings


def list_run_tree(client, run_id, verbose=0):
    """
    Returns a list of `run_id`s based on a depth-first traversal of a run tree. See: #get_run_tree

    :param client: mlflow.tracking.MlflowClient
        The MlflowClient to use
    :param run_id: str
        The `run_id` of the root run
    :param verbose:
        `verbose=1` prints a detailed progress report
    :return: list[str]
    """
    tree = get_run_tree(client, run_id, include_runs=False, verbose=verbose)
    run_id_list = []

    def traverse(run_id, node):
        for child_id, child in node["children"].items():
            traverse(child_id, child)
        run_id_list.append(run_id)

    traverse(*list(tree.items())[0])
    return run_id_list


def get_run_tree(client, run_id, include_runs=True, verbose=0):
    """
    Loads a tree of runs. The parent/child relationship is based on the `parent` tag in child runs which holds the
    parent's `run_id`.

    :param client: mlflow.tracking.MlflowClient
        The MlflowClient to use
    :param run_id: str
        The `run_id` of the root run
    :param include_runs:
        Whether to include the run objects
    :param verbose:
        `verbose=1` prints a detailed progress report
    :return: dict
         A `dict` representing the run tree `{"<run_id>: {"run": <run object>, "children":{ ... }}"}`
    """

    run = client.get_run(run_id)
    exp_id = run.info.experiment_id

    if verbose > 0:
        print("Collecting run tree:")
        print("*", f"{run.info.run_id} ({run.data.tags['stage']})" if "stage" in run.data.tags else run.info.run_id)

    run_dict = {
        "run": run if include_runs else None,
        "children": {}
    }
    tree = {run_id: run_dict}

    def collect_children(parent_id, parent_dict, level=1):

        # recursion
        children = list(client.search_runs(exp_id, filter_string=f"tags.parent='{parent_id}'"))
        for i, child in enumerate(children):

            if verbose > 0:
                run_desc = f"{child.info.run_id} ({child.data.tags['stage']})" \
                    if "stage" in child.data.tags else child.info.run_id
                print("  " * level, "*", f"{i + 1}/{len(children)}:", run_desc)

            child_dict = {
                "run": child if include_runs else None,
                "children": {}
            }
            parent_dict["children"][child.info.run_id] = child_dict

            collect_children(child.info.run_id, child_dict, level=level + 1)

    collect_children(run_id, run_dict)
    return tree


def get_stage(client, experiment_id, stage, recursive=False, finished=True, verbose=0):
    """
    Loads all runs of a stage based on the tag `stage`. This method can either return a list of runs or a dict
    of trees based on the parent-tree structure of the stage.

    :param client: mlflow.tracking.MlflowClient
        The MlflowClient to use
    :param experiment_id: str
        Id of the experiment holding the stage
    :param stage: str
        Name of the stage
    :param recursive:
        Whether to load a list of runs (`recursive=False`) or a dict with the parent-tree structure
    :param finished:
        Whether to load only finished runs
    :param verbose:
        `verbose=1` prints a detailed progress report
    :return:
        Either a list of runs (`recursive=False`) or a dict with the parent-tree structure
    """

    if verbose > 0:
        print(f"Start loading stage: '{stage}'")

    runs_stage = client.search_runs(experiment_id, f"tags.stage='{stage}'")

    if verbose > 0:
        print(f"Done loading stage: '{stage}'")

    if not recursive:
        return [r for r in runs_stage if not finished or r.info.status == "FINISHED"]
    else:

        print(f"Loading tree")

        def build_tree(children_dict):

            # build level dict
            level_dict = {}
            for child_id, child_dict in children_dict.items():
                if "parent" in child_dict["run"].data.tags:
                    # get or set parent dict
                    parent_dict = level_dict.setdefault(
                        child_dict["run"].data.tags["parent"],
                        {"run": None, "children": {}})

                    # add child to parent dict
                    parent_dict["children"][child_dict["run"].info.run_id] = child_dict

            #                 elif child_dict is not None:
            #                     raise Exception(f"Missing parent tag for '{r.info.run_id}'")
            #                 else:
            #                     break

            # recurse
            if len(level_dict) == 0:

                return children_dict

            else:

                # get parent runs
                parent_runs = [client.get_run(parent_id) for parent_id in level_dict.keys()]

                # filter parent runs if applicable
                if finished:
                    parent_runs = [r for r in parent_runs if r.info.status == "FINISHED"]

                # set parent runs:
                for r in parent_runs:
                    level_dict[r.info.run_id]["run"] = r

                # recurse
                return build_tree(level_dict)

        return build_tree({r.info.run_id: {"run": r, "childen": {}} for r in runs_stage})


def delete_run_recursively(client, run_id, dry_run=True, verbose=0):
    """
    Deletes a run and all it's children. For this function, the parent/children relationship is defined
    by the tag `parent` in child runs. The value of the `parent` tag is the `run_id` of the parent.

    :param client: mlflow.tracking.MlflowClient
        The MlflowClient to use
    :param run_id: str
        The run id to delete
    :param dry_run:
        Whether to do a dry run (instead of deleting)
    :param verbose:
        `verbose=1` prints a detailed progress report
    :return:
        A tuple with ids: (collected ids, "successfully" deleted ids)
    """

    if dry_run:
        warnings.warn("Dry run: nothing is deleted! Set `dry_run=False` to actually delete things.")

    ids_to_delete = list_run_tree(client, run_id, verbose=verbose)

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
