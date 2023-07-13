import os

import event_model
import napari
import numpy as np
from IPython import get_ipython

examples_dir = os.path.dirname(__file__)
get_ipython().run_line_magic("run", f"-i {examples_dir}/prepare_basler_env.py")

# Basler/ophyd:
emulated_basler_camera = BaslerCamera(cam_num=0, verbose=True, name="basler_cam")  # noqa: F821

# Generate images:
ny, nx = emulated_basler_camera.image_shape.get()
WGB = get_wandering_gaussian_beam(nf=256, nx=nx, ny=ny, seed=6313448000)  # noqa: F821
emulated_basler_camera.set_custom_images(WGB)
emulated_basler_camera.exposure_time.put(10)  # in [ms]

# Napari:
napari_viewer = napari.Viewer()
napari_viewer.text_overlay.visible = True
napari_viewer_layer = napari_viewer.add_image(np.zeros((ny, nx), dtype=float), rgb=False)
napari_viewer_layer.contrast_limits = [
    emulated_basler_camera.pixel_level_min.get(),
    emulated_basler_camera.pixel_level_max.get(),
]
napari_viewer.text_overlay.text = f"Image shape: {(ny, nx)}"

filler = event_model.Filler({"BASLER_CAM_HDF5": BaslerCamHDF5Handler})  # noqa: F821

current_frame = None


def update_viewer(name, doc):
    global current_frame
    # Swap the image data in for the reference (e.g. filepath) in the documents.
    name, doc = filler(name, doc)
    if name == "datum":
        current_frame = doc["datum_kwargs"]["frame"]
    if (name == "event") and ("basler_cam_image" in doc["data"]):
        # Update napari
        image = doc["data"]["basler_cam_image"]
        napari_viewer_layer.data = image
        napari_viewer.text_overlay.text = f"Image #{current_frame}: shape={image.shape}"
        napari_viewer.reset_view()


RE.subscribe(update_viewer)  # noqa: F821

# (uid,) = RE(bp.count([emulated_basler_camera], num=1000))
# data = emulated_basler_camera._viewer_layer.data
# print(f"{data = }, {data.shape = }, {data.dtype = }")
