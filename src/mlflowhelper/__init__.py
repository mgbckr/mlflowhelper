# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution

from .tracking import (get_loading_information, log_vars, managed_artifact,
                       managed_artifact_dir, set_experiment, set_load,
                       set_skip_log, set_tracking_uri, start_run)

__all__ = [
    "get_loading_information", "get_loading_information", "log_vars", "managed_artifact",
    "managed_artifact_dir", "set_experiment", "set_load",
    "set_skip_log", "set_tracking_uri", "start_run"]

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
