import logging
import os
from datetime import datetime

import bluesky.plans as bp  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np  # noqa: F401
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree

import ophyd_basler
from ophyd_basler.basler_camera import BaslerCamera  # noqa: F401
from ophyd_basler.basler_handler import BaslerCamHDF5Handler
from ophyd_basler.custom_images import get_wandering_gaussian_beam  # noqa: F401
from ophyd_basler.utils import configure_logger, logger_basler, plot_images  # noqa: F401

plt.ion()

os.environ["PYLON_CAMEMU"] = "1"

root_dir = "/tmp/basler"
_ = make_dir_tree(datetime.now().year, base_path=root_dir)

RE = RunEngine({})

db = Broker.named("local")
db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)
RE.subscribe(db.insert)

bec = BestEffortCallback()
RE.subscribe(bec)

# from bluesky.utils import ts_msg_hook
# RE.msg_hook = ts_msg_hook

device_metadata, devices = ophyd_basler.available_devices()
print(device_metadata)

configure_logger(logger_basler, log_level=logging.INFO)
