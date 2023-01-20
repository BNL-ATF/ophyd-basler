import h5py
from area_detector_handlers.handlers import HandlerBase


class BaslerCamHDF5Handler(HandlerBase):
    specs = {"BASLER_CAM_HDF5"}

    def __init__(self, filename):
        self._name = filename

    def __call__(self, frame):
        with h5py.File(self._name, "r") as f:
            entry = f["/entry/image"]
            return entry[frame]
