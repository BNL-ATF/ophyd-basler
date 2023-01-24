import os
from datetime import datetime

import bluesky.plans as bp  # noqa F401
import numpy as np  # noqa F401
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree

import ophyd_basler
from ophyd_basler.basler_camera import BaslerCamera  # noqa F401
from ophyd_basler.basler_handler import BaslerCamHDF5Handler

os.environ["PYLON_CAMEMU"] = "1"

root_dir = "/tmp/basler"
_ = make_dir_tree(datetime.now().year, base_path=root_dir)

RE = RunEngine({})

db = Broker.named("local")
db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)
RE.subscribe(db.insert)

bec = BestEffortCallback()
RE.subscribe(bec)

device_metadata, devices = ophyd_basler.available_devices()
print(device_metadata)
