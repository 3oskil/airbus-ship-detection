import os
import yaml
import numpy as np

from keras.utils import plot_model
from PIL import Image, ImageDraw, ImageFont


def load_config(path) -> dict:
    """
    Loads configuration settings from a YAML file.

    Parameters:
        path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing configuration settings.
    """
    with open(path, 'r', encoding='utf-8') as yml_file:
        config_ = yaml.safe_load(yml_file)
    return config_


def get_model(config):
    """
    Retrieves a model based on the configuration.

    Parameters:
        config (dict): Configuration dictionary with key 'model_name' indicating which model to build.

    Returns:
        keras.Model: The specified neural network model.
    """
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
    """
    Creates a new directory for the model run, ensuring no overwrite occurs by incrementing a run number.

    Parameters:
        base_path (str): The base path for the directory, without the run number.

    Returns:
        str: The path to the newly created directory.
    """
    run_number = 1
    full_path = f'{base_path}_{run_number}'

    while os.path.exists(full_path):
        run_number += 1
        full_path = f'{base_path}_{run_number}'

    os.makedirs(full_path)
    print(f'Folder created: {full_path}')
    return full_path


def save_model_plot(run_path, model_name, model):
    """
    Saves a plot of the model architecture to the specified directory.

    Parameters:
        run_path (str): The directory where the model plot should be saved.
        model_name (str): The name of the model, used to name the plot file.
        model (keras.Model): The model to be plotted.
    """
    path_model_plot = f'{run_path}/{model_name}_model_plot.jpg'
    plot_model(model, path_model_plot, dpi=46)
    print('Model plot saved to:', path_model_plot)


def overlay_mask_on_image(image, mask):
    """
    Creates an overlay of a mask on an image.

    Parameters:
        image (np.ndarray or PIL.Image.Image): The original image.
        mask (np.ndarray): The mask to overlay, should be the same size as the image.

    Returns:
        PIL.Image.Image: The image with the mask overlay.
    """
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
    """
    Adds text to an image.

    Parameters:
        image (PIL.Image.Image): The image to add text to.
        text (str): The text to add.

    Returns:
        PIL.Image.Image: The image with text added.
    """
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
    """
    Calculates the total number of pixels represented by a run-length encoding (RLE).

    Parameters:
        rle (np.ndarray): The RLE encoded mask as a 2D numpy array with [start, length] pairs.

    Returns:
        int: The total number of pixels in the RLE mask.
    """
    if rle.size > 0:
        return np.sum(rle[:, 1])
    return 0


def encoded_pixels2rle(encoded_pixels):
    """
    Converts an encoded pixels string to a run-length encoding (RLE) numpy array.

    Parameters:
        encoded_pixels (str): The encoded pixels string.

    Returns:
        np.ndarray: The RLE as a 2D numpy array with [start, length] pairs.
    """
    if isinstance(encoded_pixels, str):
        return np.array(list(zip(*[iter(int(x) for x in encoded_pixels.split())] * 2)))
    return np.array([])


def object_pixels(encoded_pixels):
    """
    Calculates the number of pixels in an object based on its encoded pixels string.

    Parameters:
        encoded_pixels (str): The encoded pixels string of the object.

    Returns:
        int: The number of pixels in the object.
    """
    return rle_pixels(encoded_pixels2rle(encoded_pixels))


def rle2mask(rle, shape=(768, 768)):
    """
    Converts a run-length encoding (RLE) into a mask.

    Parameters:
        rle (np.ndarray): The RLE as a 2D numpy array with [start, length] pairs.
        shape (tuple): The shape of the mask (height, width).

    Returns:
        np.ndarray: The mask as a 2D numpy array.
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    run_ranges = [(start - 1, start + length - 1) for (start, length) in rle]

    for a, b in run_ranges:
        mask[a:b] = 1

    return mask.reshape(shape).T


def get_combined_masks(img_id, df):
    """
    Combines all RLE masks for a given image ID into a single mask.

    Parameters:
        img_id (str): The image ID.
        df (pd.DataFrame): The dataframe containing the RLE masks.

    Returns:
        np.ndarray: The combined mask as a 2D numpy array.
    """
    return rle2mask(encoded_pixels2rle(' '.join(df[df.ImageId == img_id]['EncodedPixels'].fillna('').astype(str))))


def read_image(img_id, img_dir):
    """
    Reads an image from a directory given its ID.

    Parameters:
        img_id (str): The image ID.
        img_dir (str): The directory containing the image.

    Returns:
        PIL.Image.Image: The image.
    """
    return Image.open(os.path.join(img_dir, img_id))


def iou(mask1, mask2):
    """
    Computes the Intersection over Union (IoU) between two masks.

    Parameters:
        mask1 (np.ndarray): The first mask.
        mask2 (np.ndarray): The second mask.

    Returns:
        float: The IoU between the two masks.
    """
    inter = np.sum((mask1 >= 0.5) & (mask2 >= 0.5))
    union = np.sum((mask1 >= 0.5) | (mask2 >= 0.5))
    return inter / (1e-8 + union - inter)


def f_score(tp, fn, fp, beta=2.):
    """
    Calculates the F-beta score.

    Parameters:
        tp (int): The number of true positives.
        fn (int): The number of false negatives.
        fp (int): The number of false positives.
        beta (float): The beta value of the F-score.

    Returns:
        float: The F-beta score.
    """
    if tp + fn + fp < 1:
        return 1.
    num = (1 + beta ** 2) * tp
    return num / (num + (beta ** 2) * fn + fp)


def confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh=0.5):
    """
    Counts true positives, false negatives, and false positives based on IoU thresholding.

    Parameters:
        predict_mask_seq (list): The list of predicted masks.
        truth_mask_seq (list): The list of ground truth masks.
        iou_thresh (float): The IoU threshold to consider a detection as true positive.

    Returns:
        tuple: The counts of true positives, false negatives, and false positives.
    """
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
    """
    Calculates the mean F-score over a set of IoU thresholds.

    Parameters:
        predict_mask_seq (list): The list of predicted masks.
        truth_mask_seq (list): The list of ground truth masks.
        iou_thresholds (list, optional): The IoU thresholds to average over. Defaults to a range from 0.5 to 0.95.
        beta (float): The beta value of the F-score.

    Returns:
        float: The mean F-score.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    return np.mean([f_score(tp, fn, fp, beta) for (tp, fn, fp) in
                    [confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh) for iou_thresh in iou_thresholds]])


def get_masks(img_id, df):
    """
    Retrieves all masks for a given image ID from a dataframe.

    Parameters:
        img_id (str): The image ID.
        df (pd.DataFrame): The dataframe containing the masks.

    Returns:
        list: A list of masks for the given image ID.
    """
    return [rle2mask(encoded_pixels2rle(encoded_pixels)) for encoded_pixels in
            df[df.ImageId == img_id]['EncodedPixels']]


def get_obj_count(img_id, df):
    """
    Counts the number of objects (masks) for a given image ID in a dataframe.

    Parameters:
        img_id (str): The image ID.
        df (pd.DataFrame): The dataframe containing the masks.

    Returns:
        int: The number of objects for the given image ID.
    """
    return df[df.ImageId == img_id]['EncodedPixels'].count()


def preprocess_image(image_path, img_size):
    """
    Preprocesses an image for model input.

    Parameters:
        image_path (str): The path to the image.
        img_size (tuple): The target size (width, height) to resize the image.

    Returns:
        np.ndarray: The preprocessed image array suitable for model input.
    """
    image = Image.open(image_path)
    image_resized = image.resize(img_size)
    image_array = np.array(image_resized) / 255.0  # Normalize if your model expects normalized inputs
    return np.expand_dims(image_array, axis=0)  # Add batch dimension
