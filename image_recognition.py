import numpy as np
from skimage import exposure
import base64
from PIL import Image, ImageOps, ImageChops
from io import BytesIO


def replace_transparent_background(image):
    # image_arr = np.array(image)
    has_no_alpha = len(image.shape) < 3 or image.shape[2] < 4
    if has_no_alpha:
        return image

    alpha1 = 0
    r2, g2, b2, alpha2 = 255, 255, 255, 255

    red, green, blue, alpha = image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]
    mask = (alpha == alpha1)
    image[:, :, :4][mask] = [r2, g2, b2, alpha2]

    print(image[:, :, :3])

    return Image.fromarray(image.astype(np.uint8))


def trim_borders(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    
    return image


def pad_image(image):
    return ImageOps.expand(image, border=30, fill='#fff')


def resize_image(image):
    return image.resize((8, 8), Image.LINEAR)


def invert_colors(image):
    return ImageOps.invert(image)


def scale_down_intensity(image):
    image_arr = np.array(image)
    image_arr = exposure.rescale_intensity(image_arr, out_range=(0, 16))
    return Image.fromarray(image_arr)

def to_grayscale(image):
    return image.convert('L')


def process_image(data_uri):
    image = replace_transparent_background(data_uri)

    is_empty = not image.getbbox()
    if is_empty:
        return None

    # image = replace_transparent_background(image)
    image = trim_borders(image)
    image = pad_image(image)
    image = to_grayscale(image)
    image = invert_colors(image)
    image = resize_image(image)
    image = scale_down_intensity(image)

    return np.array([
        np.array(image).flatten()
    ])