# mlflowhelper


A set of tools for working with *mlflow* (see https://mlflow.org).

## Features

* managed artifact logging and **loading** enables
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

```


#### Central logging and loading behavior management

Logging and loading behavior can be managed in a central way:
```python

``` 


### Easy parameter logging

*mlflowhelper* helps you to never forget logging parameters again by making it easy to log all existing variables
using `mlflowhelper.log_vars`.

```python
import mlflowhelper

def main(param1, param2, param3="defaultvalue", verbose=0, *args, **kwargs):
    some_variable = "x"
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



## TODOs / Possible features
* check if loading works across experiments
* fine-grained loading behavior (right now we can only specify one run as a source for all artifacts)
* purge local artifacts (check via API which runs are marked as deleted and delete their artifacts)


## Note
This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
