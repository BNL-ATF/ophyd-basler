import ophyd_basler
from ophyd_basler.basler_camera import BaslerCamera
from ophyd_basler.basler_handler import BaslerCamHDF5Handler

import os
import numpy as np
import bluesky.plans as bp
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree
from datetime import datetime

os.environ['PYLON_CAMEMU'] = "1"

root_dir = '/tmp/basler'
_ = make_dir_tree(datetime.now().year, base_path=root_dir)

RE = RunEngine({})

db = Broker.named('local')
db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)
RE.subscribe(db.insert)

bec = best_effort.BestEffortCallback()
RE.subscribe(bec)

device_metadata, devices = ophyd_basler.available_devices()
print(device_metadata)
