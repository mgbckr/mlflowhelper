class DagManager:

    def __init__(self, executor=None):
        self.all_nodes = {}
        self.root_nodes = {}
        self.params = {}

    def add(self, func, alias=None, parents=None, parameters=None, repeat=1, parent_mode_cartesian=True):

        if alias is None:
            alias = func.__name__

        print("Adding:", alias)

        # parse parents
        parents_alias = None
        if parents is not None:
            parents_alias = []
            for parent in parents:
                if not isinstance(parent, str):
                    parents_alias.append(parent.__name__)
                else:
                    parents_alias.append(parent)

        # define node
        node = {
            "func": func,
            "alias": alias,
            "parents": {},
            "children": {},
            "parameters": parameters,
            "repeat": repeat,
            "stats": {
                "runs": []
            }
        }

        # check if alias is already known
        if alias in self.all_nodes:
            raise Exception(f"Alias already known: {alias}")

        # check parents
        if parents is not None:
            consistent_repeat = None
            for parent_alias in parents_alias:
                if parent_alias not in self.all_nodes:
                    raise Exception(f"Parent not known: {parent_alias}")
                else:
                    if not parent_mode_cartesian:
                        parent_repeat = self.all_nodes[parents_alias]["repeat"]
                        if consistent_repeat is None:
                            consistent_repeat = parent_repeat
                        elif parent_repeat is not consistent_repeat:
                            raise Exception(f"Parent repeats not consistent ({consistent_repeat} != {parent_repeat})")

        # add new node to dict of all nodes
        self.all_nodes[alias] = node

        # register nodes with other nodes (either as root or in parents)
        if parents is not None:
            for parent_alias in parents_alias:
                self.all_nodes[parent_alias]["children"][alias] = node
                node["parents"][alias] = self.all_nodes[parent_alias]
        else:
            self.root_nodes[alias] = node

    def set_param(self, key, value):
        self.params[key] = value

    def set_params(self, params):
        self.params = {**self.params, **params}

    def run(self):

        # build execution path
        execution_path = []
        queue = []
        for alias, node in self.root_nodes.items():
            queue.append(node)

        # run nodes
        for node in execution_path:
            pass
            # with mlflow.start_run():
            #     def traverse():
            #         pass


if __name__ == '__main__':

    def stage1_1(p=1):
        return 12, 13

    def stage1_2(p=1):
        return 14

    def stage2(stage1_1_results, stage1_2_results, x=99):
        twelve, thirteen = stage1_1_results
        return 99

    dag = DagManager()
    dag.add(stage1_1, repeat=5)
    dag.add(stage1_2, repeat=2)
    dag.add(stage2, parents=[stage1_1, stage1_2])
    dag.run()
