# Airbus Ship Detection Challenge

The Airbus Ship Detection Challenge, hosted on Kaggle, is aimed at advancing the development of machine learning models
capable of detecting ships in satellite images. This competition provides participants with a dataset of satellite
imagery, challenging them to create algorithms that can accurately identify and localize ships. The goal is to leverage
state-of-the-art machine learning and computer vision techniques to improve maritime safety and monitoring.

For more information, visit the [competition page](https://www.kaggle.com/competitions/airbus-ship-detection).

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

It is recommended to use a virtual environment for this project to manage dependencies. Create a virtual environment and
activate it using:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Then, install the required packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Configuration

Before running any training, validation, or testing, you need to fill out the config.yaml file located in the configs
folder according to your requirements.

### Training

To start the training, validation, and testing process, run the following command:

```bash
python main.py --path_config_file configs/config.yaml
```

This process is expected to take about half a day with the default settings in the configuration file and an RTX 4090
GPU.

If you encounter any issues with dataset preprocessing, remove the created folder in the datasets directory and try
again.

After completing the execution of main.py, a runs folder will be created containing a subfolder named after the model
and run number. Inside this folder, you will find:

- a tests folder with masks and overlayed images from the test split.
- best model weights.
- the filled config file used for the run.
- images of plots showing metrics and loss per epoch.
- a plot of the model architecture.
- images of some predictions on the validation set.

### Inference

For running inference with a trained model, use the following command format:

```bash
python inference.py --path_config_file --model_path --image_path --output_path
```

Example:

```bash
python inference.py --path_config_file runs/unet_v1_1/config.yaml --model_path runs/unet_v1_1/best_unet_v1.hdf5 --image_path datasets/airbus_ship/test/images/9037fff8d.jpg --output_path test_inference.jpg
```