from ophyd_basler.basler_camera import BaslerCamera

import os
import numpy as np
import bluesky.plans as bp

import pytest

def test_emulated_basler_camera(RE, db, make_dirs):

    os.environ['PYLON_CAMEMU'] = "1"

    emulated_basler_camera = BaslerCamera(cam_num=0)

    uid, = RE(bp.count([emulated_basler_camera], num=3))

    hdr = db[uid]

    assert np.array(list(hdr.data(field='basler_cam_image', fill=True))).shape == (3, 1040, 1024)

