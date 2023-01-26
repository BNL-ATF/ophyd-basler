import logging
from logging import StreamHandler

import matplotlib.pyplot as plt

logger_basler = logging.getLogger("basler")


def configure_logger(logger, log_level=logging.DEBUG, handlers=[StreamHandler]):
    log_file_format = "[%(levelname)1.1s %(asctime)s %(name)s %(module)s:%(lineno)d] %(message)s"
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    for handler_class in handlers:
        handler = handler_class()
        handler.setFormatter(logging.Formatter(fmt=log_file_format))
        logger.addHandler(handler)
        handler.setLevel(log_level)


def plot_images(data, nrows=None, ncols=None, save_path=None):
    """
    Usage
    -----

        nrows, ncols = 2, 4
        uid, = RE(bp.scan([<detector>], <motor>, <start>, <stop>, nrows * ncols))
        hdr = db[uid]
        data = np.array(list(hdr.data("<field_name>")))
        plot_images(data, nrows=nrows, ncols=ncols)

    """
    if nrows is None or ncols is None:
        nrows = data.shape[0]
        ncols = 1

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    for row in range(nrows):
        print(f"{row = }")
        for col in range(ncols):
            print(f"  {col = } --> {ax[row][col]}")
            ax[row][col].imshow(data[row * ncols + col], vmin=data.min(), vmax=data.max())

    if save_path is not None:
        plt.savefig(save_path)
