import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from utils.utils import overlay_mask_on_image


def test_model(model, model_path, test_data, num_batches, test_folder):
    """
    Loads a trained model's weights, makes predictions on the test dataset, and saves the predicted masks
    and overlay images to a specified folder.

    Parameters:
        model (keras.models.Model): The trained model to evaluate.
        model_path (str): Path to the trained model weights.
        test_data (keras.utils.Sequence): A batched data generator for the test dataset.
        num_batches (int): The number of batches to process from the test dataset.
        test_folder (str): The folder path where the prediction results will be saved.
    """
    model.load_weights(model_path)

    os.makedirs(test_folder, exist_ok=True)

    print('Making predictions...')
    for batch in range(num_batches):
        print(f'Batch number {batch + 1}')
        x_batch, _ = test_data[batch]
        predictions = model.predict(x_batch)

        for i, pred_mask in tqdm(enumerate(predictions)):
            original_image = (x_batch[i] * 255).astype(np.uint8)
            mask_filename = f'mask_{batch}_{i}.jpg'
            overlay_filename = f'overlay_{batch}_{i}.jpg'

            pred_mask_image = Image.fromarray((pred_mask.squeeze() * 255).astype(np.uint8))
            mask_path = os.path.join(test_folder, mask_filename)
            pred_mask_image.save(mask_path)

            overlay_image = overlay_mask_on_image(original_image, pred_mask.squeeze())
            overlay_path = os.path.join(test_folder, overlay_filename)
            overlay_image.save(overlay_path)
