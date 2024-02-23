import os
import shutil
import argparse

from utils.train import train
from utils.data import get_data
from utils.test import test_model
from utils.validate import validate
from utils.utils import load_config, get_model, create_model_run_folder, save_model_plot

if __name__ == "__main__":
    # Argument parsing to accept a configuration file path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    opt = parser.parse_args()

    # Extract the path to the configuration file from the parsed options
    path_config_file = opt.path_config_file
    print('Path to config file:', path_config_file)

    # Load the model configuration from the specified YAML file
    config = load_config(path_config_file)

    # Instantiate the model based on the configuration
    model = get_model(config)
    model_name = config['model_name']  # Retrieve the model name from the configuration

    # Create a directory for the current model run, copying the configuration file into it
    run_path = create_model_run_folder(os.path.join('runs', model_name))
    shutil.copy2(path_config_file, os.path.join(run_path, os.path.basename(path_config_file)))

    # Save a plot of the model architecture in the run directory
    save_model_plot(run_path, model_name, model)

    # Define the path where the best model weights will be saved
    model_path = f'{run_path}/best_{model_name}.hdf5'

    # Prepare training, validation, and test data using the configuration
    train_data, valid_data, test_data = get_data(config)

    # Train the model using the prepared data and save the training history
    epochs = config['epochs']
    history = train(model, train_data, valid_data, epochs, model_path)

    # Perform validation using a subset of the validation data and save results
    val_num_batches = 1  # Number of validation batches to use for performance evaluation
    metrics = ['dice_score', 'true_positive_rate']  # Metrics to be calculated during validation
    validate(model, model_name, model_path, history, valid_data, val_num_batches, metrics, run_path)

    # Set up parameters for testing the model
    img_size = config['img_size']
    test_folder = f'{run_path}/tests'  # Directory to save test results
    test_num_batches = 32 // config['batch_size']  # Calculate the number of test batches
    test_data_folder = os.path.join(config['destination_dataset_path'], 'test')

    # Test the model using the specified test data and save the results
    test_model(model, model_path, test_data, test_num_batches, test_folder)
