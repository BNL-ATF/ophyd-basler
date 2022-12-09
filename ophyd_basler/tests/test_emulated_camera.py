import pytest

def test_emulated_basler_camera():

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
    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)

    db = Broker.named('temp')

    RE.subscribe(db.insert)

    emulated_basler_camera = BaslerCamera(cam_num=0)

    db.reg.register_handler("BASLER_CAM_HDF5", BaslerCamHDF5Handler, overwrite=True)

    uid, = RE(bp.count([emulated_basler_camera]))

    hdr = db[uid]

    assert np.array(list(hdr.data(field='basler_cam_image', fill=True))).shape == (1, 1040, 1024)

