import warnings

import numpy as np
from scipy.ndimage import _nd_image, _ni_support, filters, interpolation


def _extend_mode_to_code(mode):
    mode = _ni_support._extend_mode_to_code(mode)
    return mode


def zoom(
    input,
    zoom,
    output_size=None,
    output=None,
    order=3,
    mode="constant",
    cval=0.0,
    prefilter=True,
):
    """
    Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Parameters
    ----------
    input : ndarray
        The input array.
    zoom : float or sequence, optional
        The zoom factor along the axes. If a float, `zoom` is the same for each
        axis. If a sequence, `zoom` should contain one value for each axis.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    zoom : ndarray or None
        The zoomed input. If `output` is given as a parameter, None is
        returned.

    """
    if order < 0 or order > 5:
        raise RuntimeError("spline order not supported")
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError("Complex type not supported")
    if input.ndim < 1:
        raise RuntimeError("input and output rank must be > 0")
    mode = _extend_mode_to_code(mode)
    if prefilter and order > 1:
        filtered = interpolation.spline_filter(input, order, output=np.float64)
    else:
        filtered = input

    zoom = _ni_support._normalize_sequence(zoom, input.ndim)
    if output_size == None:
        output_shape = tuple(int(round(ii * jj)) for ii, jj in zip(input.shape, zoom))
        output_shape_old = tuple(int(ii * jj) for ii, jj in zip(input.shape, zoom))
        if output_shape != output_shape_old:
            warnings.warn(
                "From scipy 0.13.0, the output shape of zoom() is calculated "
                "with round() instead of int() - for these inputs the size of "
                "the returned array has changed.",
                UserWarning,
            )
    else:
        output_shape = tuple((output_size[0], output_size[1], output_size[2]))

    zoom_div = np.array(output_shape, float) - 1
    zoom = (np.array(input.shape) - 1) / zoom_div

    # Zooming to non-finite values is unpredictable, so just choose
    # zoom factor 1 instead
    zoom[~np.isfinite(zoom)] = 1

    output = _ni_support._get_output(output, input, shape=output_shape)
    zoom = np.asarray(zoom, dtype=np.float64)
    zoom = np.ascontiguousarray(zoom)
    _nd_image.zoom_shift(filtered, zoom, None, output, order, mode, cval)
    return output


def gaussian_2D(image, sigma):
    im_new = np.zeros(np.shape(image))
    for n in range(np.shape(image)[0]):
        im_new[n, :, :] = filters.gaussian_filter(image[n, :, :], sigma)
    return im_new


def resample_image(image, spacing, new_spacing, new_size=None, order=3):
    resampling_factors = tuple(o / n for o, n in zip(spacing, new_spacing))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return zoom(image, resampling_factors, new_size, order=order, mode="nearest")


def downsample_image_xy(image, spacing, new_spacing, new_size=None, order=3, sigma=1.5):
    resampling_factors = tuple(o / n for o, n in zip(spacing, new_spacing))
    image = gaussian_2D(image, sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return zoom(image, resampling_factors, new_size, order=order, mode="nearest")


def resample_mask(mask, spacing, new_spacing, new_size):
    return resample_image(mask, spacing, new_spacing, new_size, order=0)


def pad_or_crop_image(image, target_shape, fill=-1000):
    new_image = np.ones(shape=target_shape, dtype="int16") * fill

    pads = [0, 0, 0]
    crops = [0, 0, 0]
    for axis in range(3):
        if image.shape[axis] < target_shape[axis]:
            pads[axis] = (target_shape[axis] - image.shape[axis]) / 2
        elif image.shape[axis] > target_shape[axis]:
            crops[axis] = (image.shape[axis] - target_shape[axis]) / 2

    cropped = image[
        crops[0] : crops[0] + target_shape[0],
        crops[1] : crops[1] + target_shape[1],
        crops[2] : crops[2] + target_shape[2],
    ]
    new_image[
        pads[0] : pads[0] + cropped.shape[0],
        pads[1] : pads[1] + cropped.shape[1],
        pads[2] : pads[2] + cropped.shape[2],
    ] = cropped

    return new_image
