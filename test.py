import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from utils import overlay_mask_on_image


def test_model(model, model_path, test_data, num_batches, test_folder):
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
