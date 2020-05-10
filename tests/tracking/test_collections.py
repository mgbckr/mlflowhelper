
import shutil

import pytest


@pytest.fixture(autouse=True)
def cleanup_tests(request):

    def cleanup():
        shutil.rmtree("./mlruns")

    request.addfinalizer(cleanup)


def test_init_no_exception():
    import mlflowhelper.tracking.collections
    mlflowhelper.tracking.collections.MlflowDict(local_value_cache=False)


def test_set_no_exception():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_value_cache=False)
    d["a"] = "test"
    d["b"] = "test2"


def test_len():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_value_cache=False)
    d["a"] = "test"
    d["b"] = "test2"
    assert len(d) == 2


def test_get():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_value_cache=False)
    d["a"] = "test"
    assert d["a"] == "test"


def test_iter_no_exception():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_value_cache=False)
    d["a"] = "test"
    d["b"] = "test2"
    for k, v in d.items():
        pass


def test_persistence():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_value_cache=False)
    d["a"] = "test"
    d["b"] = "test2"
    del d
    d2 = mlflowhelper.tracking.collections.MlflowDict()
    assert d2["a"] == "test"


def test_meta():

    import mlflowhelper.tracking.collections
    import mlflow.entities

    d = mlflowhelper.tracking.collections.MlflowDict(local_value_cache=False)
    d["a"] = mlflowhelper.tracking.collections.MetaValue("test", mlflowhelper.tracking.collections.Meta(
        tags=("tag1", "value1"),
        params=dict(key="param1", value=99),
        metrics=[("metric1", i, i, i) for i in range(10)],
        status="FAILED"))

    assert d["a"] == "test"

    run: mlflow.entities.Run = d.get_run("a")
    assert run.data.tags["tag1"] == "value1"
    assert run.data.params["param1"] == "99"
    assert run.data.metrics["metric1"] == 9
    assert run.info.status == "FAILED"

    # update
    d["a"] = mlflowhelper.tracking.collections.MetaValue(
        "test2",
        mlflowhelper.tracking.collections.Meta(
            tags=("tag2", "value2"),
            params=dict(key="param2", value=100),
            metrics=[("metric2", i, i, i) for i in range(10)],
            status="FINISHED"),
        update=True)

    assert d["a"] == "test2"

    run: mlflow.entities.Run = d.get_run("a")

    assert run.data.tags["tag1"] == "value1"
    assert run.data.params["param1"] == "99"
    assert run.data.metrics["metric1"] == 9

    assert run.data.tags["tag2"] == "value2"
    assert run.data.params["param2"] == "100"
    assert run.data.metrics["metric2"] == 9

    assert run.info.status == "FINISHED"

    # overwrite
    d["a"] = mlflowhelper.tracking.collections.MetaValue(
        "test3",
        mlflowhelper.tracking.collections.Meta(
            tags=("tag3", "value3"),
            params=dict(key="param3", value=101),
            metrics=[("metric3", i, i, i) for i in range(10)],
            status="FINISHED"),
        update=False)

    assert d["a"] == "test3"

    run: mlflow.entities.Run = d.get_run("a")

    assert "tag1" not in run.data.tags
    assert "param1" not in run.data.params
    assert "metric1" not in run.data.metrics

    assert "tag2" not in run.data.tags
    assert "param2" not in run.data.params
    assert "metric2" not in run.data.metrics

    assert run.data.tags["tag3"] == "value3"
    assert run.data.params["param3"] == "101"
    assert run.data.metrics["metric3"] == 9
    assert run.info.status == "FINISHED"
