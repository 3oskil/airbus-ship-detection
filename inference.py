import os
import argparse
import numpy as np

from PIL import Image
from utils.utils import load_config, get_model, overlay_mask_on_image

if __name__ == "__main__":
    # Set up an argument parser
    parser = argparse.ArgumentParser()
    # Add expected command-line arguments
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    parser.add_argument('--model_path', type=str, help='Path of the model')
    parser.add_argument('--image_path', type=str, help='Path of the image')
    parser.add_argument('--output_path', type=str, help='Path of the output image')
    # Parse arguments from the command line
    opt = parser.parse_args()

    # Extract the command-line arguments
    path_config_file = opt.path_config_file
    model_path = opt.model_path
    image_path = opt.image_path
    output_path = opt.output_path

    # Load model configuration from the specified YAML file
    config = load_config(path_config_file)

    # Initialize the model based on the loaded configuration
    model = get_model(config)

    # Load pretrained weights into the model
    model.load_weights(model_path)

    # Resize and normalize the input image for prediction
    img_size = (config['img_size'], config['img_size'])
    input_image = np.expand_dims(np.array(Image.open(image_path).resize(img_size)) / 255.0, axis=0)

    # Predict the mask for the input image
    pred_mask = model.predict(input_image).squeeze()

    # Save the predicted mask as an image
    pred_mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_mask_image.save(os.path.join(os.path.dirname(output_path), 'mask_' + os.path.basename(output_path)))

    # Overlay the predicted mask on the original image and save the result
    original_image = np.array(Image.open(image_path).resize(img_size)).astype(np.uint8)
    overlay_image = overlay_mask_on_image(original_image, pred_mask)
    overlay_image.save(output_path)
