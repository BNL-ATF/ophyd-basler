import os
from datetime import datetime

import bluesky.plans as bp  # noqa F401
import numpy as np  # noqa F401
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree

from pypylon import pylon
import tempfile
import cv2

import ophyd_basler

from ophyd_basler.basler_camera import BaslerCamera  # noqa F401
from ophyd_basler.basler_handler import BaslerCamHDF5Handler

import os
os.environ["PYLON_CAMEMU"] = "1"

# hard-coded for now
n_frames = 256
width = int(1920 / 4)
height = int(1080 / 4)

length_scale = 16
quintet_bounds = np.array([[0,256],
                           [0,width],
                           [0,height],
                           [width/64, width/8],
                           [height/64, height/8]])

def gaussian(x, y, a, cx, cy, sx, sy): 
    return a * np.exp(-0.5*(np.square((x-cx)/sx) + np.square((y-cy)/sy)))

def get_wandering_gaussian_beam(nf, nx, ny, bounds, length_scale):

    ps    = np.exp(-np.square(np.fft.fftfreq(nf)*length_scale))
    data  = np.real(np.fft.ifft(ps * np.fft.fft(np.random.standard_normal(size=(5, nf)))))
    data -= data.min(axis=1)[:,None]
    data *= (bounds.ptp(axis=1) / data.ptp(axis=1))[:,None]
    data += bounds.min(axis=1)[:,None]
    
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    return gaussian(X[None,:,:], Y[None,:,:], *data[:,:,None,None])

WGB = get_wandering_gaussian_beam(256, width, height, quintet_bounds, length_scale)

# from https://github.com/basler/pypylon/issues/73:
# we customize the first camera, which is the one we'll 
# instantiate for our emulated_basler_camera ophyd device.

img_dir = tempfile.mkdtemp()

for i, image in enumerate(WGB):
    cv2.imwrite(os.path.join(img_dir,"pattern_%03d.png"%i), image)

transport_layer_factory = pylon.TlFactory.GetInstance()
device_info_list = transport_layer_factory.EnumerateDevices()
device_info = device_info_list[0]
device = transport_layer_factory.CreateDevice(device_info)
cam = pylon.InstantCamera(device)

cam.Open()

cam.ImageFilename = img_dir
cam.ImageFileMode = "On"
cam.TestImageSelector = "Off" # disable testpattern [ image file is "real-image"]
cam.PixelFormat = "Mono8" # choose one pixel format. camera emulation does conversion on the fly

cam.Close()

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

emulated_basler_camera = BaslerCamera(cam_num=0, verbose=True, root_dir=root_dir, name="basler_cam")