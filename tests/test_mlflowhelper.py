# -*- coding: utf-8 -*-

import pytest
import shutil
from matplotlib import pyplot as plt

import mlflowhelper


@pytest.fixture(autouse=True)
def cleanup_tests(request):

    def cleanup():
        shutil.rmtree("./mlruns")

    request.addfinalizer(cleanup)


def test_no_exception():
    with mlflowhelper.start_run():
        with mlflowhelper.managed_artifact("test.png") as a:
            plt.plot([1,2,3], [1,2,3]); plt.savefig(a.get_path())

