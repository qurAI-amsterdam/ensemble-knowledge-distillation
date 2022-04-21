import numpy as np
from scipy.ndimage.interpolation import rotate


def as_tuple(x, n, t=None, return_none=False):
    """Converts a value x into a tuple and ensures that the length is n"""
    if return_none and x is None:
        return None

    try:
        y = tuple(x)
        if len(y) != n:
            raise ValueError(
                f"Expected a single value or an iterable with length {n}, got {x} instead"
            )
    except TypeError:
        y = (x,) * n

    if t:
        y = tuple(t(v) for v in y)

    return y


def random_rotation(image, mask=None, sigma=10, axes=None, rotation_angle=None):
    """
    Applies random rotations to the image (which can be 2D or 3D)

    Parameters
    ----------
    image : numpy array
        2D or 3D image

    mask : numpy array or None
        2D or 3D mask of the same shape as the image. The same transformations will be applied to both image and mask.

    sigma : float or sequence of floats
        Rotation angles are drawn from a normal distribution with this standard deviation. Can be either a single scalar
        or one angle per axis

    axes : sequence of ints or None
        List of axes about which to rotate. Only relevant for 3D inputs. 0 = Sagittal, 1 = Coronal, 2 = Axial. If no list
        is supplied (None), the image is rotated about all three axes.

    Returns
    -------
    Either only the deformed image or the deformed image and the deformed mask, if a mask was supplied.
    """
    image = np.asarray(image)
    assert mask is None or (isinstance(mask, np.ndarray) and mask.shape == image.shape)
    assert image.ndim in (2, 3)

    # 2D image
    if image.ndim == 2:
        if rotation_angle is None:
            rotation_angle = np.random.normal(scale=sigma)
        deformed_image = rotate(image, rotation_angle, reshape=False, mode="reflect")
        if mask is None:
            return deformed_image
        else:
            deformed_mask = rotate(
                mask, rotation_angle, reshape=False, order=0, mode="constant", cval=0
            )
            return deformed_image, deformed_mask

    # 3D volume
    if image.ndim == 3:
        if axes is None:
            axes = [0, 1, 2]
        if rotation_angle is None:
            rotation_angle = np.random.normal(scale=sigma)
        np.random.shuffle(axes)  # over first 2 axes are rotated
        deformed_image = rotate(
            image,
            rotation_angle,
            axes=(axes[0], axes[1]),
            reshape=False,
            mode="reflect",
        )
        if mask is None:
            return deformed_image
        else:
            deformed_mask = rotate(
                mask,
                rotation_angle,
                axes=(axes[0], axes[1]),
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )
            return deformed_image, deformed_mask

    sigmas = as_tuple(sigma, len(axes), t=float)

    deformed_image = image
    deformed_mask = mask
    for axis, sigma in zip(axes, sigmas):
        rotation_axes = [i for i in range(3) if i != axis]
        if rotation_angle is None:
            rotation_angle = np.random.normal(scale=sigma)

        deformed_image = rotate(
            deformed_image,
            angle=rotation_angle,
            axes=rotation_axes,
            reshape=False,
            mode="reflect",
        )
        if mask is not None:
            deformed_mask = rotate(
                deformed_mask,
                angle=rotation_angle,
                axes=rotation_axes,
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )

    if mask is None:
        return deformed_image
    else:
        return deformed_image, deformed_mask
