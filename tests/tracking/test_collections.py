
import shutil

import pytest


@pytest.fixture(autouse=True)
def cleanup_tests(request):

    def cleanup():
        shutil.rmtree("./mlruns")

    request.addfinalizer(cleanup)


def test_init_no_exception():
    import mlflowhelper.tracking.collections
    mlflowhelper.tracking.collections.MlflowDict(local_cache=False)


def test_set_no_exception():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_cache=False)
    d["a"] = "test"
    d["b"] = "test2"


def test_len():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_cache=False)
    d["a"] = "test"
    d["b"] = "test2"
    assert len(d) == 2


def test_get():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_cache=False)
    d["a"] = "test"
    assert d["a"] == "test"


def test_iter_no_exception():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_cache=False)
    d["a"] = "test"
    d["b"] = "test2"
    for k, v in d.items():
        pass


def test_persistence():
    import mlflowhelper.tracking.collections
    d = mlflowhelper.tracking.collections.MlflowDict(local_cache=False)
    d["a"] = "test"
    d["b"] = "test2"
    del d
    d2 = mlflowhelper.tracking.collections.MlflowDict()
    assert d2["a"] == "test"
