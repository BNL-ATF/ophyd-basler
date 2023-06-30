import os
from datetime import datetime

import bluesky.plans as bp
import matplotlib.pyplot as plt
import napari
import numpy as np
import pytest
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree
from skimage import data

import ophyd_basler
from ophyd_basler.basler_camera import BaslerCamera
from ophyd_basler.basler_handler import BaslerCamHDF5Handler
from ophyd_basler.custom_images import get_wandering_gaussian_beam
from ophyd_basler.utils import plot_images

# RE, db, etc.
db = Broker.named("temp")
db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)

RE = RunEngine({})
bec = best_effort.BestEffortCallback()
bec.disable_plots()
RE.subscribe(bec)
RE.subscribe(db.insert)
root_dir = "/tmp/basler"
_ = make_dir_tree(datetime.now().year, base_path=root_dir)

# Napari:
napari_viewer = napari.Viewer()

# Basler/ophyd:
os.environ["PYLON_CAMEMU"] = "1"

print(ophyd_basler.available_devices())
emulated_basler_camera = BaslerCamera(cam_num=0, verbose=True, name="basler_cam", viewer=napari_viewer)

## Generate images:
ny, nx = emulated_basler_camera.image_shape.get()
WGB = get_wandering_gaussian_beam(nf=256, nx=nx, ny=ny, seed=6313448000)
emulated_basler_camera.set_custom_images(WGB)
emulated_basler_camera.exposure_time.put(200)

# (uid,) = RE(bp.count([emulated_basler_camera], num=100))
# data = emulated_basler_camera._viewer_layer.data
# print(f"{data = }, {data.shape = }, {data.dtype = }")
