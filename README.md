# mlflowhelper

A set of tools for working with *mlflow* (see https://mlflow.org)

## Features

* managed artifact logging and **loading**
    * **automatic** artifact logging and cleanup
    * **no overwriting files** when running scripts in parallel
    * **loading** artifact
    * **central configuration** of logging and loading behavior
* log **all** function parameters and locals with a simple call to `mlflowhelper.log_vars()`


## Documentation

### Managed artifact logging and loading

#### General functionality

```python
from matplotlib import pyplot as plt
import mlflowhelper

with mlflowhelper.start_run():
    with mlflowhelper.managed_artifact("plot.png") as artifact:
        fig = plt.figure()
        plt.plot([1,2,3], [1,2,3])
        fig.savefig(artifact.get_path())
```
This code snippet automatically logs the created artifact (`plot.png`).
At the same time if will create the artifact in a temporary folder so that you don't have to worry about
overwriting it when running your scripts in parallel.
By default, this also cleans up the artifact and the temporary folder after logging.

You can also manage artifacts on a directory level:
```python
from matplotlib import pyplot as plt
import mlflowhelper

with mlflowhelper.start_run():
    with mlflowhelper.managed_artifact_dir("plots") as artifact_dir:

        # plot 1
        fig = plt.figure()
        plt.plot([1,2,3], [1,2,3])
        fig.savefig(artifact_dir.get_path("plot1.png"))

        # plot 2
        fig = plt.figure()
        plt.plot([1,2,3], [1,2,3])
        fig.savefig(artifact_dir.get_path("plot2.png"))
```

#### Artifact loading
You may want to run experiments but reuse some precomputed artifact from a different run (such
as preprocessed data, trained models, etc.). This can be done as follows:
```python
import mlflowhelper
import pandas as pd

with mlflowhelper.start_run():
    mlflowhelper.set_load(run_id="e1363f760b1e4ab3a9e93f856f2e9341", stages=["load_data"]) # activate loading from previous run
    with mlflowhelper.managed_artifact_dir("data.csv", stage="load_data") as artifact:
        if artifact.loaded:
            # load artifact
            data = pd.read_csv(artifact.get_path())
        else:
            # create and save artifact
            data = pd.read_csv("/shared/dir/data.csv").sample(frac=1)
            data.to_csv(artifact.get_path())
```

Similarly, this works for directories of course:
```python
import mlflowhelper
import pandas as pd

mlflowhelper.set_load(run_id="e1363f760b1e4ab3a9e93f856f2e9341", stages=["load_data"]) # activate loading from previous run
with mlflowhelper.start_run():
    with mlflowhelper.managed_artifact_dir("data", stage="load_data") as artifact_dir:
        train_path = artifact_dir.get_path("test.csv")
        test_path = artifact_dir.get_path("train.csv")
        if artifact_dir.loaded:
            # load artifacts
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
        else:
            data = pd.read_csv("/shared/dir/data.csv").sample(frac=1)
            train = data.iloc[:100,:]
            test = data.iloc[100:,:]
            # save artifacts
            train.to_csv(train_path)
            test.to_csv(test_path)
```

**Note:** The `stage` parameter must be set in `mlflowhelper.managed_artifact(_dir)` to enable loading.

#### Central logging and loading behavior management

Logging and loading behavior can be managed in a central way:
```python
import mlflowhelper
import pandas as pd

with mlflowhelper.start_run():

    # activate loading the stage `load_data` from previous run `e1363f760b1e4ab3a9e93f856f2e9341`
    mlflowhelper.set_load(run_id="e1363f760b1e4ab3a9e93f856f2e9341", stages=["load_data"])

    # deactivate logging the stage `load_data`, in this case for example because it was loaded from a previous run
    mlflowhelper.set_skip_log(stages=["load_data"])

    with mlflowhelper.managed_artifact_dir("data", stage="load_data") as artifact_dir:
        train_path = artifact_dir.get_path("test.csv")
        test_path = artifact_dir.get_path("train.csv")
        if artifact_dir.loaded:
            # load artifacts
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
        else:
            data = pd.read_csv("/shared/dir/data.csv").sample(frac=1)
            train = data.iloc[:100,:]
            test = data.iloc[100:,:]
            # save artifacts
            train.to_csv(train_path)
            test.to_csv(test_path)
```

**Note:** For central managing the `stage` parameter must be set in `mlflowhelper.managed_artifact(_dir)`.


### Easy parameter logging

*mlflowhelper* helps you to never forget logging parameters again by making it easy to log all existing variables
using `mlflowhelper.log_vars`.

```python
import mlflowhelper

def main(param1, param2, param3="defaultvalue", verbose=0, *args, **kwargs):
    some_variable = "x"
    with mlflowhelper.start_run(): # mlflow.start_run() is also OK here
        mlflowhelper.log_vars(exclude=["verbose"])

if __name__ == '__main__':
    main("a", "b", something_else=6)
```
This will log:
```json
{
  "param1": "a",
  "param2": "b",
  "param3": "defaultvalue",
  "something_else": 6
}
```


### Other
There are a few more convenience functions included in `mlflowhelper`:


## TODOs / Ideas
- [ ] check if loading works across experiments
- [ ] purge local artifacts (check via API which runs are marked as deleted and delete their artifacts)
- [ ] support nested runs by creating subdirectories based on experiment and run
- [ ] support loading from central cache instead of from runs
- [ ] automatically log from where and what has been loaded
- [ ] set tags for logged stages (to check for artifacts before loading them)
- [ ] consider loading extensions:
  - [ ] does nested loading make sense (different loads for certain nested runs)?
  - [ ] does mixed loading make sense (loading artifacts from different runs for different stages)?


## Note
This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
