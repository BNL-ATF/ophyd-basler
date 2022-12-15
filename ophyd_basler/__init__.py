import pandas as pd
from ophyd import Signal
from pypylon import pylon

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


class ExternalFileReference(Signal):
    """
    A pure software Signal that describe()s an image in an external file.
    """

    def describe(self):
        resource_document_data = super().describe()
        resource_document_data[self.name].update(
            dict(
                external="FILESTORE:",
                dtype="array",
            )
        )
        return resource_document_data


def available_devices():
    """
    Returns a pandas DataFrame that outlines the devices available to pylon. This is very helpful in the field.
    """

    transport_layer_factory = pylon.TlFactory.GetInstance()
    device_info_list = transport_layer_factory.EnumerateDevices()
    device_metadata = pd.DataFrame(
        columns=["user_defined_name", "camera_model", "serial_number", "supported_formats"]
    )

    devices = []

    for device_info in device_info_list:

        device = transport_layer_factory.CreateDevice(device_info)
        camera_object = pylon.InstantCamera(device)
        devices.append(camera_object)

        user_defined_name = camera_object.GetDeviceInfo().GetUserDefinedName()
        camera_model = camera_object.GetDeviceInfo().GetModelName()
        camera_serial_number = camera_object.GetDeviceInfo().GetSerialNumber()
        supported_formats = camera_object.PixelFormat.Symbolics

        device_metadata.loc[len(device_metadata)] = (
            user_defined_name,
            camera_model,
            camera_serial_number,
            supported_formats,
        )

    return device_metadata, devices
