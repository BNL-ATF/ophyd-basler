import os
import tempfile

import cv2
import numpy as np

from ophyd_basler.utils import logger_basler as logger


def gaussian_2d(x, y, a, cx, cy, sx, sy):
    """
    Returns a two-dimensional Gaussian distribution.

    (x, y) :   input coordinates at which to evaluate the distribution
    a :        the maximum of the distribution
    (cx, cy) : the coordinates of the center of the distribution
    (sx, sy) : the spread of the distribution in each dimension
    """
    return a * np.exp(-0.5 * (np.square((x - cx) / sx) + np.square((y - cy) / sy)))


def get_wandering_gaussian_beam(nf, nx, ny, seed=0):
    """
    Generates a slowly-fluctuating Gaussian beam, and returns it in an array of
    images with shape (nf, ny, nx).
    """

    rng = np.random.default_rng(seed)

    # hard-coded for now
    time_scale = 64

    # the generated beam's parameters will stay within these bounds
    bounds = np.array([[0, 256], [0, nx], [0, ny], [nx / 64, nx / 8], [ny / 64, ny / 8]])

    # generate a Gaussian power spectrum for the beam fluctuations
    ps = np.exp(-np.square(np.fft.fftfreq(nf) * time_scale))

    # scale some Fourier-transformed white noise, and Fourier transform back to get correlated beam parameters
    beam_params = np.real(np.fft.ifft(ps * np.fft.fft(rng.standard_normal(size=(5, nf)))))

    # scale the generated data so that it varies between the specified bounds
    beam_params -= beam_params.min(axis=1)[:, None]
    beam_params *= (bounds.ptp(axis=1) / beam_params.ptp(axis=1))[:, None]
    beam_params += bounds.min(axis=1)[:, None]

    # get image coordinates
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    return gaussian_2d(X[None, :, :], Y[None, :, :], *beam_params[:, :, None, None])


def save_images(images, img_dir=None):
    if img_dir is None:
        img_dir = tempfile.mkdtemp()
    else:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)

    logger.info(f"Using '{img_dir}' to save {len(images)} images to.")
    for i, image in enumerate(images):
        cv2.imwrite(os.path.join(img_dir, "pattern_%03d.png" % i), image)
    logger.info(f"Saved {len(os.listdir(img_dir))} images into '{img_dir}'")

    return img_dir
