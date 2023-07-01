import os

import napari
from IPython import get_ipython

examples_dir = os.path.dirname(__file__)
get_ipython().run_line_magic("run", f"-i {examples_dir}/prepare_basler_env.py")

# Napari:
napari_viewer = napari.Viewer()
napari_viewer.text_overlay.visible = True

# Basler/ophyd:
emulated_basler_camera = BaslerCamera(  # noqa: F821
    cam_num=0, verbose=True, name="basler_cam", viewer=napari_viewer
)

# Generate images:
ny, nx = emulated_basler_camera.image_shape.get()
WGB = get_wandering_gaussian_beam(nf=256, nx=nx, ny=ny, seed=6313448000)  # noqa: F821
emulated_basler_camera.set_custom_images(WGB)
emulated_basler_camera.exposure_time.put(10)  # in [ms]

# (uid,) = RE(bp.count([emulated_basler_camera], num=1000))
# data = emulated_basler_camera._viewer_layer.data
# print(f"{data = }, {data.shape = }, {data.dtype = }")
