from datetime import datetime

import pytest
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree

from ophyd_basler.basler_handler import BaslerCamHDF5Handler


@pytest.fixture(scope="function")
def db():
    """
    Return a data broker
    """

    db = Broker.named("temp")
    db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)

    return db


@pytest.fixture(scope="function")
def RE(db):

    RE = RunEngine({})
    bec = best_effort.BestEffortCallback()
    bec.disable_plots()
    RE.subscribe(bec)
    RE.subscribe(db.insert)

    return RE


@pytest.fixture(scope="function")
def make_dirs():

    root_dir = "/tmp/basler"
    _ = make_dir_tree(datetime.now().year, base_path=root_dir)
