# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = mlflowhelper.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

from .  import tracking

from mlflowhelper import __version__

__author__ = "Martin Becker"
__copyright__ = "Martin Becker"
__license__ = "mit"


set_tracking_uri = tracking.set_tracking_uri
set_experiment = tracking.set_experiment

start_run = tracking.start_run
set_skip_log = tracking.set_skip_log
set_load = tracking.set_load
managed_artifact = tracking.managed_artifact
managed_artifact_dir = tracking.managed_artifact_dir

log_vars = tracking.log_vars
get_loading_information = tracking.get_loading_information

