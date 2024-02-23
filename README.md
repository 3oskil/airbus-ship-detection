# Airbus Ship Detection Challenge

## Table of Contents

- [Project description](#Project-description)
- [Notes](#Notes)
    - [To-Do](#To-Do)
- [Exploratory data analysis](#Exploratory-data-analysis)
- [Solution description](#Solution-description)
    - [Dataset-Preprocessing](#Dataset-Preprocessing)
    - [Models](#Models)
    - [Loss and metrics](#Loss-and-metrics)
    - [Training, Validation and Test](#Training,-Validation-and-Test)
- [Getting Started](#Getting-Started)
    - [Prerequisites](#Prerequisites)
    - [Installation](#Installation)
    - [Configuration](#Configuration)
    - [Training](#Training)
    - [Inference](#Inference)

## Project description

The Airbus Ship Detection Challenge, hosted on Kaggle, is aimed at advancing the development of machine learning models
capable of detecting ships in satellite images. This competition provides participants with a dataset of satellite
imagery, challenging them to create algorithms that can accurately identify and localize ships. The goal is to leverage
state-of-the-art machine learning and computer vision techniques to improve maritime safety and monitoring.

For more information, visit the [competition page](https://www.kaggle.com/competitions/airbus-ship-detection).

## Notes

1. Among the three models, only U-Net Version 1 was utilized for training.
2. The model achieved a Dice Score of approximately 0.8, a BCE-Dice Loss of 0.2, and a True Positive Rate of between
   0.75 and 0.85 for both training and validation datasets.
3. The training performance was slightly lower, attributed to the use of cross-validation with a validation set size of
   0.05, resulting in a more extensive training dataset.

### To-Do:

1. During cross-validation, create a test set.
2. Implement custom augmentation by randomly selecting boats using masks from images containing boats and inserting them
   into images without boats. This process should include resizing, rotating, and randomly placing the boats.
3. Experiment with U-Net Version 2 and U-Net++ models.
4. Integrate MLFlow and Docker for enhanced workflow management.

## Exploratory data analysis

EDA is provided in the notebook "dataset_eda.ipynb" stored in the "notebooks" folder.

## Solution description

1. #### **Dataset Preprocessing** (data.py file)
    - Dataset Validation: It checks the dataset's structure and contents, ensuring the presence of necessary directories
      and files. The validation process confirms the existence of 'train', 'val', and 'test' splits, along with their
      corresponding 'images' and 'masks' directories. It also verifies that the number of image and mask files match and
      are in the correct format.

    - Data Augmentation: Utilizes the albumentations library to perform on-the-fly data augmentation during the training
      phase. The augmentation techniques include flips, rotations, brightness and contrast adjustments, and more complex
      transformations like elastic, grid, and optical distortions. This helps improve model generalization by presenting
      a more diverse set of training examples.

    - Data Generators: Implements a SegmentationDataGenerator class that extends keras.utils.Sequence, providing a
      robust mechanism for batch-wise data feeding during model training or evaluation. This class efficiently handles
      image and mask loading, resizing, optional augmentation, and normalization. It supports shuffling to ensure
      diverse mini-batches and includes methods for visualizing batches of data, aiding in debugging and dataset
      understanding.

    - Dataset Preparation and Organization: Includes functions for organizing and preparing the dataset into a structure
      suitable for training, validation, and testing. It automates the process of copying images and masks to designated
      directories, applying data augmentation, and splitting the dataset. This setup phase ensures that the data is
      correctly partitioned and accessible for the data generators.

    - Custom Augmentation: The framework is designed to accommodate future enhancements in data augmentation techniques,
      specifically targeting the augmentation of images by introducing synthetic variations. This innovative approach
      involves extracting boat images from existing photographs using their segmentation masks, then applying
      transformations such as rotation, resizing, and random placement onto images that originally contain no boats.
      This method aims to artificially increase the diversity and complexity of the dataset by generating new, unique
      training examples. By doing so, the model can learn from a broader range of scenarios, potentially improving its
      ability to generalize across unseen data. This custom augmentation strategy is particularly valuable for tasks
      where the dataset is limited or lacks variety in certain aspects, offering a creative solution to enhance model
      performance without the need for additional real-world data.

2. #### **Models** (models.py file)

   ##### U-Net Version 1 (build_unet_v1)

   This version is a straightforward implementation of the U-Net architecture, characterized by its symmetric design
   with a contracting path to capture context and a symmetric expanding path that enables precise localization. The
   model employs conventional convolutional blocks, max pooling for downsampling, dropout for regularization, and
   transposed convolutions for upsampling. The architecture is designed to work with images of a configurable size and
   utilizes a combination of binary cross-entropy and dice loss for training, aiming to optimize both pixel-wise
   accuracy and overlap between predicted and ground truth masks.

   ##### U-Net Version 2 (build_unet_v2)

   The second version introduces Batch Normalization in each convolutional block to stabilize learning and improve
   convergence rates. The architecture follows the classic U-Net pattern but enhances feature propagation and model
   performance through the normalization layers. This version also employs ELU activation for non-linearities, aiming
   for better handling of vanishing gradient issues compared to the traditional ReLU, and includes dropout for
   regularization. The design principles remain focused on balancing feature extraction capabilities with computational
   efficiency, making it suitable for more extensive datasets or more complex segmentation tasks.

   ##### U-Net++ (build_unet_pp)

   U-Net++ introduces a sophisticated enhancement over the traditional U-Net architecture by incorporating nested, dense
   skip pathways. These modifications aim to improve the flow of information and gradients throughout the network,
   facilitating more detailed feature extraction at various scales and improving segmentation accuracy, particularly at
   boundaries and fine structures. The model uses convolutional blocks with ELU activation, dropout for regularization,
   and l2 kernel regularization to prevent overfitting. The architecture is highly configurable, allowing adjustments to
   filter sizes and layer configurations to suit different image sizes and segmentation challenges. U-Net++ is
   particularly effective in applications requiring high precision in segmentation outcomes.

   Each model is compiled with Adam optimizer, utilizing a combined loss function that includes both binary
   cross-entropy and dice loss to balance between pixel-wise classification accuracy and overlap metrics. The choice
   between these architectures offers flexibility in addressing various segmentation challenges, from basic applications
   with U-Net v1 to more complex scenarios requiring advanced features like those in U-Net v2 and U-Net++.

3. #### **Loss and metrics** (metrics.py file)

   ##### Dice Score
   The Dice score (also known as the Dice coefficient) measures the similarity between two sets, which, in the context
   of image segmentation, correspond to the predicted segmentation map and the ground truth. It ranges from 0 (no
   overlap) to 1 (perfect overlap), making it an effective metric for assessing the accuracy of segmentation models. The
   Dice score is calculated as twice the area of overlap between the predicted and true masks divided by the total
   number of pixels in both masks, with a small constant added to avoid division by zero.

   $$\text{Dice} = \frac{2 \times \sum (y_{\text{pred}} \times y_{\text{true}}) + \epsilon}{\sum y_{\text{true}} + \sum
   y_{\text{pred}} + \epsilon}$$

    - $`y_{\text{pred}}`$ - predicted segmentation map.
    - $`y_{\text{true}}`$ - ground truth segmentation map.
    - $`\sum`$ - summation over all pixels.
    - $`\epsilon`$ - a small constant (e.g., 0.0001) added to avoid division by zero.

   ##### BCE-Dice Loss

   The BCE-Dice loss combines binary cross-entropy (BCE) loss and Dice loss (1 - Dice score) into a single function.
   This hybrid approach leverages the pixel-wise classification capabilities of BCE loss and the global similarity
   measurement of Dice loss, providing a balanced optimization criterion that encourages the model to improve both local
   accuracy and overall shape alignment with the ground truth. By summing the BCE loss and the Dice loss, this combined
   loss function helps mitigate the limitations of using either loss individually, promoting better performance in
   segmentation tasks, especially when dealing with imbalanced datasets or irregular object shapes.

   $$\text{TPR} = \frac{\sum (y_{\text{true}} \times \text{round}(y_{\text{pred}}))}{\sum y_{\text{true}}}$$

    - $`y_{\text{true}}`$ - ground truth segmentation map.
    - $`\text{round}(y_{\text{pred}})`$ - predicted segmentation map rounded to the nearest integer (0 or 1).
    - Other symbols as defined previously.

   ##### True Positive Rate

   The True Positive Rate (TPR), also known as sensitivity or recall, quantifies the proportion of actual positives (
   true conditions) correctly identified by the model. In segmentation models, it measures how well the model identifies
   pixels or regions that genuinely belong to the object of interest. The TPR is particularly important in medical image
   analysis or other applications where missing a relevant feature can have significant consequences. It is calculated
   by dividing the number of true positive predictions (pixels correctly classified as belonging to the target class) by
   the total number of actual positives in the ground truth.

   $$\text{BCE-Dice Loss} = \text{BCE}(y_{\text{true}}, y_{\text{pred}}) + (1 - \text{Dice})$$

    - $`\text{BCE}(y_{\text{true}}, y_{\text{pred}})`$ - predicted segmentation map.
    - $`1 - \text{Dice}`$ - ground truth segmentation map.

4. #### **Training, Validation and Test** (train.py, validate.py, test.py file)

   The **train** function orchestrates the model training process using provided training and validation datasets. It
   incorporates several key components:

   **Callbacks** for enhancing training:
    - **ModelCheckpoint** saves the best model based on validation loss.
    - **ReduceLROnPlateau** reduces the learning rate when a metric has stopped improving, helping to fine-tune the
      model.
    - **EarlyStopping** halts training when a monitored metric stops improving, preventing overfitting.

   The model is trained over a specified number of epochs, with training and validation data fed into the model.

   **Validation** is twofold, involving both quantitative and qualitative evaluations:
    - Quantitative: The plot_history function generates plots for loss and each metric over the training epochs, helping
      to visually assess the model's learning progress and performance on both training and validation data.
    - Qualitative: The validate_model function performs predictions on the validation set and saves images comparing
      true masks (ground truth) with predicted masks. This visual comparison provides intuitive insights into the
      model's segmentation capabilities.

   The **test** evaluates the model on a separate test dataset. It loads the best model weights, makes predictions, and
   saves both the predicted masks and overlay images, offering a final qualitative assessment of the model's
   generalization ability on unseen data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing
purposes.

### Prerequisites

- Python 3.10.12

### Installation

1. **Dataset Preparation**

   Download the dataset from the Airbus Ship Detection Challenge on Kaggle at the following URL:

   https://www.kaggle.com/competitions/airbus-ship-detection/data

   After downloading, unzip the dataset in the root folder of this project.

2. **Virtual Environment**

   It is recommended to use a virtual environment for this project to manage dependencies. Create a virtual environment
   and activate it using:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   Then, install the required packages using the requirements.txt file:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Before running main.py or inference.py you need to fill out the config.yaml file located in the
folder "configs" according to your requirements.

### Training

To start the training, validation, and testing process, run the following command:

   ```bash
   python main.py --path_config_file configs/config.yaml
   ```

This process is expected to take about half a day with default settings in the configuration file and the NVIDIA GeForce
RTX 4090 GPU.

If you encounter any issues with dataset preprocessing, remove the created folder in the datasets directory and try
again.

After completing the execution of main.py, folder "runs" will be created containing a sub-folder named model_name from
config.yaml and run number. Inside this folder, you will find:

- folder "tests" with masks and overlaid images from the test split;
- best weights of a model;
- the filled configuration file used for the run;
- images of plots showing metrics and losses over epochs;
- an image of the model architecture;
- images with comparison of some predicted and true masks from the validation set.

### Inference

For running inference with a trained model, use the following command format:

```bash
python inference.py --path_config_file --model_path --image_path --output_path
```

Example:

```bash
python inference.py --path_config_file runs/unet_v1_1/config.yaml --model_path runs/unet_v1_1/best_unet_v1.hdf5 --image_path datasets/airbus_ship/test/images/9037fff8d.jpg --output_path test_inference.jpg
```