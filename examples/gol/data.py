import jax
import jax.numpy as jnp
from matplotlib import image as mpimg


def get_blinker_4x4():
    return jnp.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )


def get_blinker_10x10():
    return jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


def get_blinker_n(n: int):
    init = jnp.zeros((n, n), dtype=int)
    return init.at[n - 2, n - 2 : n + 1].set(1)


def get_wizards_logo():
    """Load wizards image and convert to binary pattern."""
    import os

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(assets_dir, "assets", "wizards.jpg")
    img = jnp.array(mpimg.imread(img_path))

    # Convert to grayscale
    if len(img.shape) == 3:
        gray_img = jnp.mean(img[:, :, :3], axis=2)
    else:
        gray_img = img

    # Apply threshold
    bw = jnp.where(gray_img < 128, 1, 0)

    # Resize to 512x512 maintaining aspect ratio
    height, width = bw.shape[:2]
    if height > width:
        new_height = 512
        new_width = int(512 * width / height)
    else:
        new_width = 512
        new_height = int(512 * height / width)

    resized = jax.image.resize(bw, (new_height, new_width), method="nearest")

    # Pad to make square
    pad_y = (512 - new_height) // 2
    pad_x = (512 - new_width) // 2

    pad_top = pad_y
    pad_bottom = 512 - new_height - pad_top
    pad_left = pad_x
    pad_right = 512 - new_width - pad_left

    square = jnp.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    assert square.shape == (512, 512)
    return square


def get_small_wizards_logo(size=128):
    """Get a downsampled version of the wizards logo.

    Note: For sizes larger than 512, will upscale from the 512x512 original.
    """
    full_logo = get_wizards_logo()
    small_logo = jax.image.resize(full_logo, (size, size), method="nearest")
    return jnp.where(small_logo > 0.5, 1, 0)  # Re-binarize after resize
