# -*- coding: utf-8 -*-

import pytest
import mlflowhelper
from matplotlib import pyplot as plt


def test_no_exception():
    with mlflowhelper.start_run():
        with mlflowhelper.managed_artifact("test.png") as a:
            plt.plot([1,2,3], [1,2,3]); plt.savefig(a.get_path())