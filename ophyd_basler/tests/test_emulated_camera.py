import os

import bluesky.plans as bp
import numpy as np

import ophyd_basler
from ophyd_basler.basler_camera import BaslerCamera


def test_emulated_basler_camera(RE, db, make_dirs):

    os.environ["PYLON_CAMEMU"] = "1"

    print(ophyd_basler.available_devices())

    emulated_basler_camera = BaslerCamera(cam_num=0, verbose=True, name="basler_cam")
    (uid,) = RE(bp.count([emulated_basler_camera], num=3))
    hdr = db[uid]
    assert np.array(list(hdr.data(field="basler_cam_image", fill=True))).shape == (3, 1040, 1024)
