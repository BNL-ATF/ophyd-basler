from ophyd_basler.basler_camera import BaslerCamera
from ophyd_basler.basler_handler import BaslerCamHDF5Handler

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
bec = best_effort.BestEffortCallback()
RE.subscribe(bec)

db = Broker.named('temp')

RE.subscribe(db.insert)

cam = BaslerCamera()

db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)

uid, = RE(bp.count([cam]))

hdr = db[uid]

print(np.array(list(hdr.data(field='basler_cam_image', fill=True))).shape)

