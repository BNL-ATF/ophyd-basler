import datetime
import itertools
import os
import warnings
from collections import deque
from pathlib import Path

import h5py
import numpy as np
from event_model import compose_resource
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.sim import NullStatus, new_uid
from pypylon import pylon

from . import ExternalFileReference
from .custom_images import save_images
from .utils import logger_basler as logger


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

    def __init__(
        self,
        *args,
        root_dir="/tmp/basler",
        cam_num=0,
        pixel_format="Mono8",
        trigger_mode="Off",
        verbose=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._root_dir = root_dir
        self._cam_num = cam_num
        self._pixel_format = pixel_format
        self._trigger_mode = trigger_mode
        self._verbose = verbose

        # Used for the emulated cameras only.
        self._img_dir = None

        # Resource/datum docs related variables.
        self._asset_docs_cache = deque()
        self._resource_document = None
        self._datum_factory = None

        transport_layer_factory = pylon.TlFactory.GetInstance()
        device_info_list = transport_layer_factory.EnumerateDevices()
        self.device_info = device_info_list[self._cam_num]
        self.device = transport_layer_factory.CreateDevice(self.device_info)
        self.camera_object = pylon.InstantCamera(self.device)

        # Temporarily open the camera to read the metadata
        self.camera_object.Open()

        self.user_defined_name.put(self.camera_object.GetDeviceInfo().GetUserDefinedName())
        self.camera_model.put(self.camera_object.GetDeviceInfo().GetModelName())
        self.serial_number.put(self.camera_object.GetDeviceInfo().GetSerialNumber())
        self.image_shape.put((self.camera_object.Height(), self.camera_object.Width()))
        self.pixel_level_min.put(self.camera_object.PixelDynamicRangeMin())
        self.pixel_level_max.put(self.camera_object.PixelDynamicRangeMax())
        self.active_format.put(self.camera_object.PixelFormat.GetValue())
        self.payload_size.put(self.camera_object.PayloadSize())

        self.camera_object.TriggerMode.SetValue(self._trigger_mode)
        self.camera_object.PixelFormat.SetValue(self._pixel_format)

        self.camera_object.Close()

        if self._verbose:
            print(f"User-defined camera name    : {self.user_defined_name.get()}")
            print(f"Camera model                : {self.camera_model.get()}")
            print(f"Camera serial number        : {self.serial_number.get()}")
            print(f"Image shape (height, width) : {self.image_shape.get()} pixels")
            print(f"Pixel format                : {self.active_format.get()}")
            print(f"Camera min. pixel level     : {self.pixel_level_min.get()}")
            print(f"Camera max. pixel level     : {self.pixel_level_max.get()}")
            print(f"Grab timeout                : {self.grab_timeout.get()} ms")
            print(f"Trigger mode                : {self._trigger_mode}")
            print(f"GigE transport payload size : {self.payload_size.get():,} bytes")

    def set_custom_images(self, images=None, img_dir=None):
        """
        Set custom images for the emulated camera either via an ndarray or a
        directory with images.

        Parameters
        ----------
        images : ndarray
            an ndarray of the image data the images shaped as (num_frames, ny, nx)
        img_dir : str
            a directory name with a series of image files.
        """

        if images is None and img_dir is None:
            raise ValueError(
                "Either the 'images' kwarg should be used to "
                "specify an array of images, or the 'img_dir' kwarg "
                "with the existing directory of images should be "
                "passed."
            )
        if images is not None:
            img_dir = save_images(images)

        elif img_dir is not None:
            logger.info(f"Using '{img_dir}' with the existing {len(os.listdir(img_dir))} images.")

        self._img_dir = img_dir

        self.camera_object.Open()
        self.camera_object.ImageFilename = img_dir
        self.camera_object.ImageFileMode = "On"
        self.camera_object.TestImageSelector = "Off"  # disable testpattern [image file is "real-image"]
        self.camera_object.PixelFormat = (
            "Mono8"  # choose one pixel format; camera emulation does conversion on the fly
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

        logger.debug(f"grabbed a frame with the shape {image.shape}")
        logger.debug(f"{np.where(image.max()) = }  |  {image.max() = }")
        return image

    def trigger(self):

        logger.debug("started trigger")

        super().trigger()
        logger.debug("started grabbing")
        image = self.grab_image()

        logger.debug("finisihed grabbing")
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

        logger.debug("finisihed trigger")

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

        if self.camera_object.DeviceInfo.GetModelName() == "Emulation":
            # This setting makes sure we continue our iteration over the set of
            # predefined images on each trigger.
            self.camera_object.AcquisitionMode.SetValue("SingleFrame")

        # Exposure time can't be less than self.camera_object.ExposureTime.Min.
        # We use seconds for ophyd, and microseconds for pylon:
        if not self.camera_object.ExposureTimeAbs == 1e3 * self.exposure_time.get():

            if self._verbose:
                logger.debug(f"Setting exposure time to {self.exposure_time.get()} ms")

            min_exposure_us = self.camera_object.ExposureTimeAbs.Min
            max_exposure_us = self.camera_object.ExposureTimeAbs.Max

            # If the requested value is less than the minimum exposure time, use the minimum exposure time.
            if min_exposure_us > 1e3 * self.exposure_time.get():
                self.exposure_time.put(1e-3 * min_exposure_us)
                warnings.warn(
                    f"Desired exposure time ({1e3 * self.exposure_time.get()} us) is less than "
                    f"the minimum exposure time ({min_exposure_us} us). Proceeding with minimum exposure time."
                )

            # If the requested value is greater than the maximum exposure time, use the maximum exposure time.
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
