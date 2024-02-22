import os
import yaml
import numpy as np

from keras.utils import plot_model
from PIL import Image, ImageDraw, ImageFont


def load_config(path) -> dict:
    with open(path, 'r', encoding='utf-8') as yml_file:
        config_ = yaml.safe_load(yml_file)
    return config_


def get_model(config):
    if config['model_name'] == 'unet_v1':
        from utils.models import build_unet_v1
        model = build_unet_v1(config)
    elif config['model_name'] == 'unet_v2':
        from utils.models import build_unet_v2
        model = build_unet_v2(config)
    elif config['model_name'] == 'unet_pp':
        from utils.models import build_unet_pp
        model = build_unet_pp(config)
    else:
        model = None
    return model


def create_model_run_folder(base_path):
    run_number = 1
    full_path = f'{base_path}_{run_number}'

    while os.path.exists(full_path):
        run_number += 1
        full_path = f'{base_path}_{run_number}'

    os.makedirs(full_path)
    print(f'Folder created: {full_path}')
    return full_path


def save_model_plot(run_path, model_name, model):
    path_model_plot = f'{run_path}/{model_name}_model_plot.jpg'
    plot_model(model, path_model_plot, dpi=46)
    print('Model plot saved to:', path_model_plot)


def overlay_mask_on_image(image, mask):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    mask = np.where(mask > 0.5, 1, 0)

    red_intensity = 255
    green_blue_intensity = 0
    red_channel = (mask * red_intensity).astype(np.uint8)
    green_channel = (mask * green_blue_intensity).astype(np.uint8)
    blue_channel = (mask * green_blue_intensity).astype(np.uint8)
    mask_rgb = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    mask_rgb_image = Image.fromarray(mask_rgb)
    if image.size != mask_rgb_image.size:
        mask_rgb_image = mask_rgb_image.resize(image.size, Image.ANTIALIAS)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    overlay_image = Image.blend(image, mask_rgb_image, alpha=0.3)

    return overlay_image


def add_text_to_image(image, text):
    """Adds text label on top of an image."""
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    text_width = draw.textlength(text, font=font)
    text_height = font.size

    new_height = image.height + text_height + 10
    new_image = Image.new('RGB', (image.width, new_height), "white")
    new_image.paste(image, (0, text_height + 10))

    draw = ImageDraw.Draw(new_image)
    text_x = (new_image.width - text_width) / 2
    draw.text((text_x, 5), text, fill="black", font=font)

    return new_image


def rle_pixels(rle):
    """ returns: the pixel count in the object encoded by 'rle' """
    if rle.size > 0:
        return np.sum(rle[:, 1])
    return 0


def encoded_pixels2rle(encoded_pixels):
    if isinstance(encoded_pixels, str):
        return np.array(list(zip(*[iter(int(x) for x in encoded_pixels.split())] * 2)))
    return np.array([])


def object_pixels(encoded_pixels):
    """ returns: the number of pixels in the object encoded by 'encoded_pixels' """
    return rle_pixels(encoded_pixels2rle(encoded_pixels))


def rle2mask(rle, shape=(768, 768)):
    """
    rle: 2D numpy array with rows of form [start, run-length]
    shape: (rows, cols) the shape of the referenced image
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    run_ranges = [(start - 1, start + length - 1) for (start, length) in rle]

    for a, b in run_ranges:
        mask[a:b] = 1

    return mask.reshape(shape).T


def get_combined_masks(img_id, df):
    return rle2mask(encoded_pixels2rle(' '.join(df[df.ImageId == img_id]['EncodedPixels'].fillna('').astype(str))))


def read_image(img_id, img_dir):
    return Image.open(os.path.join(img_dir, img_id))


def iou(mask1, mask2):
    inter = np.sum((mask1 >= 0.5) & (mask2 >= 0.5))
    union = np.sum((mask1 >= 0.5) | (mask2 >= 0.5))
    return inter / (1e-8 + union - inter)


def f_score(tp, fn, fp, beta=2.):
    if tp + fn + fp < 1:
        return 1.
    num = (1 + beta ** 2) * tp
    return num / (num + (beta ** 2) * fn + fp)


def confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh=0.5):
    predict_masks = [m for m in predict_mask_seq if np.any(m >= 0.5)]
    truth_masks = [m for m in truth_mask_seq if np.any(m >= 0.5)]

    if len(truth_masks) == 0:
        tp, fn, fp = 0.0, 0.0, float(len(predict_masks))
        return tp, fn, fp

    pred_hits = np.zeros(len(predict_masks), dtype=bool)  # 0 miss, 1 hit
    truth_hits = np.zeros(len(truth_masks), dtype=bool)  # 0 miss, 1 hit

    for p, pred_mask in enumerate(predict_masks):
        for t, truth_mask in enumerate(truth_masks):
            if iou(pred_mask, truth_mask) > iou_thresh:
                truth_hits[t] = True
                pred_hits[p] = True

    tp = np.sum(pred_hits)
    fn = len(truth_masks) - np.sum(truth_hits)
    fp = len(predict_masks) - tp

    return tp, fn, fp


def mean_f_score(predict_mask_seq, truth_mask_seq, iou_thresholds=None, beta=2.):
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    return np.mean([f_score(tp, fn, fp, beta) for (tp, fn, fp) in
                    [confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh) for iou_thresh in iou_thresholds]])


def get_masks(img_id, df):
    return [rle2mask(encoded_pixels2rle(encoded_pixels)) for encoded_pixels in
            df[df.ImageId == img_id]['EncodedPixels']]


def get_obj_count(img_id, df):
    return df[df.ImageId == img_id]['EncodedPixels'].count()


def preprocess_image(image_path, img_size):
    image = Image.open(image_path)
    image_resized = image.resize(img_size)
    image_array = np.array(image_resized) / 255.0  # Normalize if your model expects normalized inputs
    return np.expand_dims(image_array, axis=0)  # Add batch dimension
