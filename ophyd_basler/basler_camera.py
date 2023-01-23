import datetime
import itertools
import logging
import os
import tempfile
import warnings
from collections import deque
from pathlib import Path

import cv2
import h5py
import numpy as np
from event_model import compose_resource
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.sim import NullStatus, new_uid
from pypylon import pylon

from . import ExternalFileReference

logger = logging.getLogger("basler")


class BaslerCamera(Device):

    image = Cpt(ExternalFileReference, kind="normal")
    mean = Cpt(Signal, kind="hinted")
    exposure_time = Cpt(Signal, value=1000, kind="config")  # exposure time, in milliseconds
    user_defined_name = Cpt(Signal, kind="config")
    camera_model = Cpt(Signal, kind="config")
    serial_number = Cpt(Signal, kind="config")
    image_shape = Cpt(Signal, kind="config")
    pixel_level_min = Cpt(Signal, kind="config")
    pixel_level_max = Cpt(Signal, kind="config")
    active_format = Cpt(Signal, kind="config")
    payload_size = Cpt(Signal, kind="config")
    grab_timeout = Cpt(Signal, value=5000, kind="config")

    def __init__(self, *args, root_dir="/tmp/basler", cam_num=0, pixel_format="Mono8", verbose=False, **kwargs):
        super().__init__(*args, **kwargs)

        self._root_dir = root_dir

        self._asset_docs_cache = deque()
        self._resource_document = None
        self._datum_factory = None

        self.pixel_format = pixel_format
        self.verbose = verbose

        transport_layer_factory = pylon.TlFactory.GetInstance()
        device_info_list = transport_layer_factory.EnumerateDevices()
        self.device_info = device_info_list[cam_num]
        self.device = transport_layer_factory.CreateDevice(self.device_info)
        self.camera_object = pylon.InstantCamera(self.device)

        # temporarily open the camera to read the metadata
        self.camera_object.Open()

        self.user_defined_name.put(self.camera_object.GetDeviceInfo().GetUserDefinedName())
        self.camera_model.put(self.camera_object.GetDeviceInfo().GetModelName())
        self.serial_number.put(self.camera_object.GetDeviceInfo().GetSerialNumber())
        self.image_shape.put((self.camera_object.Height(), self.camera_object.Width()))
        self.pixel_level_min.put(self.camera_object.PixelDynamicRangeMin())
        self.pixel_level_max.put(self.camera_object.PixelDynamicRangeMax())
        self.active_format.put(self.camera_object.PixelFormat.GetValue())
        self.payload_size.put(self.camera_object.PayloadSize())

        # these are hardcoded for now, but should make them more flexible in the future
        trigger_mode = "Off"

        self.camera_object.TriggerMode.SetValue(trigger_mode)
        self.camera_object.PixelFormat.SetValue(self.pixel_format)

        self.camera_object.Close()

        if self.verbose:

            print("User-defined camera name    :", self.user_defined_name.get())
            print("Camera model                :", self.camera_model.get())
            print("Camera serial number        :", self.serial_number.get())
            print("Image shape (height, width) :", self.image_shape.get(), "pixels")
            print("Pixel format                :", self.active_format.get())
            print("Camera min. pixel level     :", self.pixel_level_min.get())
            print("Camera max. pixel level     :", self.pixel_level_max.get())
            print("Grab timeout                :", self.grab_timeout.get(), "ms")
            print("Trigger mode                :", trigger_mode)
            print("GigE transport payload size : " + "{:,}".format(self.payload_size.get()) + " bytes")

    def set_custom_images(self, images):

        img_dir = tempfile.mkdtemp()
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(img_dir, "pattern_%03d.png" % i), image)

        self.camera_object.Open()

        self.camera_object.ImageFilename = img_dir
        self.camera_object.ImageFileMode = "On"
        self.camera_object.TestImageSelector = "Off"  # disable testpattern [image file is "real-image"]
        self.camera_object.PixelFormat = (
            "Mono8"  # choose one pixel format. camera emulation does conversion on the fly
        )

        self.camera_object.Close()

    def grab_image(self):

        self.camera_object.StartGrabbingMax(1)

        while self.camera_object.IsGrabbing():
            with self.camera_object.RetrieveResult(
                self.grab_timeout.get(), pylon.TimeoutHandling_ThrowException
            ) as res:

                if res.GrabSucceeded():
                    image = np.array(res.Array)
                else:
                    raise Exception("Could not grab image with pylon")

        self.camera_object.StopGrabbing()

        # print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} grabbed a frame with pylon")
        return image

    def trigger(self):

        if self.verbose:
            logger.debug(f"started trigger @ {datetime.datetime.now().isoformat()}")

        super().trigger()
        image = self.grab_image()

        if self.verbose:
            logger.debug(f"finisihed trigger @ {datetime.datetime.now().isoformat()}")

        logger.debug(f"original shape: {image.shape}")

        current_frame = next(self._counter)
        self._dataset.resize((current_frame + 1, *self.image_shape.get()))
        logger.debug(f"{self._dataset = }\n{self._dataset.shape = }")
        self._dataset[current_frame, :, :] = image

        datum_document = self._datum_factory(datum_kwargs={"frame": current_frame})
        self._asset_docs_cache.append(("datum", datum_document))

        self.image.put(datum_document["datum_id"])
        self.mean.put(image.mean())

        super().trigger()
        return NullStatus()

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
            Path(self._resource_document["root"]) / Path(self._resource_document["resource_path"])
        )

        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        logger.debug(f"{self._data_file = }")

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *self.image_shape.get())),
            maxshape=(None, *self.image_shape.get()),
            chunks=(1, *self.image_shape.get()),
            dtype="float64",
            compression="lzf",
        )
        self._counter = itertools.count()

        self.camera_object.Open()

        # Exposure time can't be less than self.camera_object.ExposureTime.Min.
        # We use seconds for ophyd, and microseconds for pylon:
        if not self.camera_object.ExposureTimeAbs == 1e3 * self.exposure_time.get():

            if self.verbose:
                logger.debug(f"Setting exposure time to {self.exposure_time.get()} ms")

            min_exposure_us = self.camera_object.ExposureTimeAbs.Min
            max_exposure_us = self.camera_object.ExposureTimeAbs.Max

            # If the requested value is less than the minimum exposure time, use the minimum exposure time
            if min_exposure_us > 1e3 * self.exposure_time.get():
                self.exposure_time.put(1e-3 * min_exposure_us)
                warnings.warn(
                    f"Desired exposure time ({1e3 * self.exposure_time.get()} us) is less than "
                    f"the minimum exposure time ({min_exposure_us} us). Proceeding with minimum exposure time."
                )

            elif max_exposure_us < 1e3 * self.exposure_time.get():
                self.exposure_time.put(1e-3 * max_exposure_us)
                warnings.warn(
                    f"Desired exposure time ({1e3 * self.exposure_time.get()} us) is greater than "
                    f"the maximum exposure time ({max_exposure_us} us). Proceeding with maximum exposure time."
                )

            else:
                self.camera_object.ExposureTimeAbs.SetValue(1e3 * self.exposure_time.get())

    def unstage(self):

        self.camera_object.Close()
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
