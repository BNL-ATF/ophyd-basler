import os

import bluesky.plans as bp
import numpy as np
import pytest

import ophyd_basler
from ophyd_basler.basler_camera import BaslerCamera
from ophyd_basler.custom_images import get_wandering_gaussian_beam
from ophyd_basler.utils import plot_images


@pytest.mark.parametrize("exposure_ms", [200, 2000])
def test_emulated_basler_camera(RE, db, make_dirs, exposure_ms, num_counts=8):
    os.environ["PYLON_CAMEMU"] = "1"

    print(ophyd_basler.available_devices())

    emulated_basler_camera = BaslerCamera(cam_num=0, verbose=True, name="basler_cam")

    ny, nx = emulated_basler_camera.image_shape.get()
    WGB = get_wandering_gaussian_beam(nf=256, nx=nx, ny=ny, seed=6313448000)

    emulated_basler_camera.set_custom_images(WGB)
    emulated_basler_camera.exposure_time.put(exposure_ms)

    (uid,) = RE(bp.count([emulated_basler_camera], num=num_counts))

    hdr = db[uid]
    tbl = hdr.table()

    first_duration = tbl["time"][1].timestamp() - hdr.start["time"]
    other_durations = np.array(tbl["time"].diff(), dtype=float)[1:] / 1e9
    durations = np.array([first_duration, *other_durations])
    print(f"{durations = }")

    # convert to seconds for comparison with bluesky
    assert (np.median(durations) > 1e-3 * exposure_ms) and (np.median(durations) < 1e-3 * exposure_ms + 0.1)

    images = np.array(list(hdr.data(field="basler_cam_image", fill=True)))
    print(images)

    assert images.shape == (num_counts, 1040, 1024)

    plot_images(
        images, ncols=4, nrows=2, save_path=f"/tmp/test_emulated_basler_camera_exposure_ms={exposure_ms:d}.png"
    )
