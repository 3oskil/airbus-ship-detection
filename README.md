# Airbus Ship Detection Challenge

The Airbus Ship Detection Challenge, hosted on Kaggle, is aimed at advancing the development of machine learning models
capable of detecting ships in satellite images. This competition provides participants with a dataset of satellite
imagery, challenging them to create algorithms that can accurately identify and localize ships. The goal is to leverage
state-of-the-art machine learning and computer vision techniques to improve maritime safety and monitoring.

For more information, visit the [competition page](https://www.kaggle.com/competitions/airbus-ship-detection).

## Exploratory data analysis

EDA is provided in the notebook "dataset_eda.ipynb" stored in the "notebooks" folder.

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