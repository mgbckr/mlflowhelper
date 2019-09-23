import warnings


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
        Either the ids to delete (`dry_run=True`) OR the deleted ids (`dry_run=False`)
    """

    run = client.get_run(run_id)
    exp_id = run.info.experiment_id

    if dry_run:
        warnings.warn("Dry run: nothing is deleted! Set `dry_run=False` to actually delete things.")

    if verbose > 0:
        print("Collecting run ids to delete:")
        print("*", f"{run.info.run_id} ({run.data.tags['stage']})" if "stage" in run.data.tags else run.info.run_id)

    ids_to_delete = []

    def collect_ids(parent_id, level=1):

        # recursion
        children = list(client.search_runs(exp_id, filter_string=f"tags.parent='{parent_id}'"))
        for i, child in enumerate(children):
            if verbose > 0:
                run_desc = f"{child.info.run_id} ({child.data.tags['stage']})" \
                    if "stage" in child.data.tags else child.info.run_id
                print("  " * level, "*", f"{i + 1}/{len(children)}:", run_desc)
            collect_ids(child.info.run_id, level=level + 1)

        # recursion end
        ids_to_delete.append(parent_id)

    collect_ids(run_id)

    if not dry_run:

        if verbose > 0:
            print("Deleting ids:")

        for id_to_delete in ids_to_delete:
            if verbose > 0:
                print("*", id_to_delete)

            run_to_delete = client.get_run(id_to_delete)
            if run_to_delete.info.lifecycle_stage == "active":
                client.delete_run(id_to_delete)
            else:
                if verbose > 0:
                    print("  Run id '{}' skipped since it is not 'active' (lifecycle_stage={}).".format(
                        id_to_delete, run_to_delete.info.lifecycle_stage))

    return ids_to_delete
