# -*- coding: utf-8 -*-

import shutil

import mlflowhelper
import pytest
from matplotlib import pyplot as plt


@pytest.fixture(autouse=True)
def cleanup_tests(request):

    def cleanup():
        shutil.rmtree("./mlruns")

    request.addfinalizer(cleanup)


def test_no_exception():
    with mlflowhelper.start_run():
        with mlflowhelper.managed_artifact("test.png") as a:
            plt.plot([1, 2, 3], [1, 2, 3])
            plt.savefig(a.get_path())
