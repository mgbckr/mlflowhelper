# mlflowhelper


A set of tools for working with *mlflow* (see https://mlflow.org).

## Features

* managed artifact logging and **loading** enables
    * run scripts in parallel without worrying about overwriting files
    * automatic artifact logging
    * artifact **loading**
    * central configuration of logging and loading behavior
* easy parameter logging enables to log function parameters and locals with a simple call to `mlflowhelper.log_vars()`


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
This code snippet will automatically create a temporary folder for you. 
Then using the path provided by `artifact.get_path()` you automatically write to that temporary folder.
The created artifact will automatically be logged.
After the artifact is logged, it's local copy is deleted automatically to preserve space.
When the run is finished, the temporary directory is deleted as well. 

This also works for directories:
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

#### Central logging and loading behavior management


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
TBA



Note
====

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
