import datetime
import h5py
import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from . import ExternalFileReference
from .basler_handler import read_shadow_file

from pypylon import pylon
from pathlib import Path
from collections import deque
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.sim import NullStatus, new_uid
from area_detector_handlers.handlers import HandlerBase
from event_model import compose_resource

os.environ['PYLON_CAMEMU'] = "1"

logger = logging.getLogger("basler")

class BaslerCamera(Device):

    image    = Cpt(ExternalFileReference, kind="normal")
    mean     = Cpt(Signal, kind="hinted")
    shape    = Cpt(Signal, kind="normal")
    exposure = Cpt(Signal, value=10, kind="config")

    def __init__(self, name='basler_cam', root_dir='/tmp/basler', **kwargs): # where should root_dir be?
        super().__init__(name=name, **kwargs)

        self._root_dir = root_dir

        self._asset_docs_cache = deque()
        self._resource_document = None
        self._datum_factory = None

        transport_layer_factory = pylon.TlFactory.GetInstance()
        device_info_list = transport_layer_factory.EnumerateDevices()
        print(device_info_list)

        number_of_devices = len(device_info_list)
        print(f"{number_of_devices = }")

        self.device_info = device_info_list[0]
        self.device = transport_layer_factory.CreateDevice(self.device_info)
        self.camera_object = pylon.InstantCamera(self.device)

        self.camera_object.Open()

        self.user_defined_name = self.camera_object.GetDeviceInfo().GetUserDefinedName()
        self.camera_model = self.camera_object.GetDeviceInfo().GetModelName()
        self.camera_serial_no = self.camera_object.GetDeviceInfo().GetSerialNumber()
        self.width = self.camera_object.Width()
        self.height = self.camera_object.Height()
        self.pixel_level_min = self.camera_object.PixelDynamicRangeMin()
        self.pixel_level_max = self.camera_object.PixelDynamicRangeMax()
        self.active_format = self.camera_object.PixelFormat.GetValue()
        self.formats_supported = self.camera_object.PixelFormat.Symbolics
        self.payload_size = self.camera_object.PayloadSize()
        self.grab_timeout = 5000 

        self._frame_shape = (self.height, self.width)
        self.camera_object.Close()

    def grab_images(self):

        self.camera_object.StartGrabbingMax(self.exposure.get())
        counter = itertools.count()
        images  = []

        while self.camera_object.IsGrabbing():
            grab_result = self.camera_object.RetrieveResult(self.grab_timeout, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = grab_result.Array
                current_frame = next(counter)
                images.append(image)

            grab_result.Release()

        print(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} "
                f"grabbed {len(images)} frames"
                )

        self.camera_object.StopGrabbing()
        return np.array(images)

    def trigger(self, verbose=True):

        super().trigger()
        images = self.grab_images()
        self.camera_object.Close()

        logger.debug(f"original shape: {images.shape}")
        # Averaging over all frames and summing 3 RGB channels
        averaged = images.mean(axis=0)

        current_frame = next(self._counter)
        self._dataset.resize((current_frame + 1, *self._frame_shape))
        logger.debug(f"{self._dataset = }\n{self._dataset.shape = }")
        self._dataset[current_frame, :, :] = averaged

        datum_document = self._datum_factory(datum_kwargs={"frame": current_frame})
        self._asset_docs_cache.append(("datum", datum_document))

        self.image.put(datum_document["datum_id"])
        self.mean.put(averaged.mean())

        super().trigger()
        return NullStatus()

    def update_components(self, image):

        self.image.put(image)
        self.shape.put(image.shape)
        self.mean.put(image.mean())

    def stage(self):

        super().stage()
        date = datetime.datetime.now()
        self._assets_dir = date.strftime("%Y/%m/%d")
        data_file = f"{new_uid()}.h5"

        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="BASLER_CAM_HDF5",
            root=self._root_dir,
            resource_path=str(Path(self._assets_dir) / Path(data_file)),
            resource_kwargs={},
        )

        self._data_file = str(
            Path(self._resource_document["root"])
            / Path(self._resource_document["resource_path"])
        )

        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        logger.debug(f"{self._data_file = }")

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset("averaged",
                                             data=np.full(fill_value=np.nan,
                                                          shape=(1, *self._frame_shape)),
                                             maxshape=(None, *self._frame_shape),
                                             chunks=(1, *self._frame_shape),
                                             dtype="float64",
                                             compression="lzf")
        self._counter = itertools.count()

        self.camera_object.Open()

        trigger_mode = "Off"

        print("User-defined camera name    :", self.user_defined_name)
        print("Camera model                :", self.camera_model)
        print("Camera serial number        :", self.camera_serial_no)
        print("Image width  (X, ncols)     :", self.width, "pixels")
        print("Image height (Y, nrows)     :", self.height, "pixels")
        print("Pixel format                :", self.active_format)
        print("Camera min. pixel level     :", self.pixel_level_min)
        print("Camera max. pixel level     :", self.pixel_level_max)
        print("Grab timeout                :", self.grab_timeout, "ms")
        print("Trigger mode                :", trigger_mode)
        print("GigE transport payload size : " + "{:,}".format(self.payload_size) + " bytes")
        print("\nCamera supported pixel formats:\n", self.formats_supported)

        self.camera_object.TriggerMode.SetValue(trigger_mode)
        desired_pixel_format = "Mono8"
        self.camera_object.PixelFormat.SetValue(desired_pixel_format)

    def unstage(self):
        super().unstage()
        del self._dataset
        self._h5file_desc.close()
        self._resource_document = None
        self._datum_factory = None

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item

