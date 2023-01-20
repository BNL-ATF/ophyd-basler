import os

import bluesky.plans as bp
import numpy as np
import pytest

from examples import prepare_emulated_basler_env

import ophyd_basler
from ophyd_basler.basler_camera import BaslerCamera


@pytest.mark.parametrize("exposure_seconds", [0.2, 2])
def test_emulated_basler_camera(RE, db, make_dirs, exposure_seconds, num_counts=5):
    os.environ["PYLON_CAMEMU"] = "1"

    print(ophyd_basler.available_devices())

    emulated_basler_camera = BaslerCamera(cam_num=0, verbose=True, name="basler_cam")

    emulated_basler_camera.exposure_time.put(exposure_seconds)

    (uid,) = RE(bp.count([emulated_basler_camera], num=num_counts))

    hdr = db[uid]
    tbl = hdr.table()

    first_duration = tbl["time"][1].timestamp() - hdr.start["time"]
    other_durations = np.array(tbl["time"].diff(), dtype=float)[1:] / 1e9
    durations = np.array([first_duration, *other_durations])
    print(f"{durations = }")

    assert (durations > exposure_seconds).all() and (durations < exposure_seconds + 0.1).all()

    images = np.array(list(hdr.data(field="basler_cam_image", fill=True)))
    print(images)

    assert images.shape == (num_counts, 1040, 1024)
