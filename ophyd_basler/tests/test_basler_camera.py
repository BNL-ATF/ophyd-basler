from ophyd_basler.basler_camera import BaslerCamera
from ophyd_basler.basler_handler import BaslerCamHDF5Handler

import numpy as np
import bluesky.plans as bp
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker

os.environ['PYLON_CAMEMU'] = "1"

RE = RunEngine({})
bec = best_effort.BestEffortCallback()
RE.subscribe(bec)

db = Broker.named('temp')

RE.subscribe(db.insert)

cam = BaslerCamera()

db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)

uid, = RE(bp.count([cam]))

hdr = db[uid]

print(np.array(list(hdr.data(field='basler_cam_image', fill=True))).shape)

