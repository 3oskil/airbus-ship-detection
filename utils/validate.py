import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from utils.utils import overlay_mask_on_image, add_text_to_image


def plot_history(history, metrics, path):
    """
    Plots the training and validation loss and metrics.

    Parameters:
    - history: The history object returned by the fit method of a model.
    - metrics: List of strings, names of the metrics that were tracked.
    - path: Directory where to save the plots.
    """
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss - BCE_DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.jpg'))
    plt.close()

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Metric - {metric}')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(path, f'{metric}.jpg'))
        plt.close()


def validate_model(model, model_path, valid_data, num_batches, runs_path):
    model.load_weights(model_path)

    print('Making predictions...')
    for batch in range(num_batches):
        x_batch, y_batch = valid_data[batch]
        predictions = model.predict(x_batch)
        num_predictions = predictions.shape[0]
        mini_batch_size = 4

        for mini_batch in range(0, num_predictions, mini_batch_size):
            image_path = os.path.join(runs_path, f'val_pred_comparison_{mini_batch}.jpg')

            x_mini_batch = (x_batch[mini_batch:mini_batch + mini_batch_size] * 255).astype(np.uint8)
            y_mini_batch = y_batch[mini_batch:mini_batch + mini_batch_size]

            pred_mini_batch = predictions[mini_batch:mini_batch + mini_batch_size]

            mini_batch_images = []
            for original_image, mask, pred_mask in zip(x_mini_batch, y_mini_batch, pred_mini_batch):
                original_overlay_image = overlay_mask_on_image(original_image, mask.squeeze())
                predicted_overlay_image = overlay_mask_on_image(original_image, pred_mask.squeeze())

                original_overlay_image = add_text_to_image(original_overlay_image, 'True Mask')
                predicted_overlay_image = add_text_to_image(predicted_overlay_image, 'Pred Mask')

                total_width = original_overlay_image.width + predicted_overlay_image.width
                max_height = max(original_overlay_image.height, predicted_overlay_image.height)
                combined_image = Image.new('RGB', (total_width, max_height))
                combined_image.paste(original_overlay_image, (0, 0))
                combined_image.paste(predicted_overlay_image, (original_overlay_image.width, 0))

                mini_batch_images.append(combined_image)

            grid_width = mini_batch_images[0].width * 2
            grid_height = mini_batch_images[0].height * 2
            final_image = Image.new('RGB', (grid_width, grid_height))

            final_image.paste(mini_batch_images[0], (0, 0))
            final_image.paste(mini_batch_images[1], (mini_batch_images[0].width, 0))
            final_image.paste(mini_batch_images[2], (0, mini_batch_images[0].height))
            final_image.paste(mini_batch_images[3], (mini_batch_images[0].width, mini_batch_images[0].height))

            final_image.save(image_path)


def validate(model, model_name, model_path, history, valid_data, num_batches, metrics, run_path):
    print(f'Validating {model_name}...')
    plot_history(history, metrics, run_path)
    print('Plots for loss and metrics values over epochs for train and valid were saved.')
    validate_model(model, model_path, valid_data, num_batches, run_path)
    print('Images with true and predicted masks were saved.')
