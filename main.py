import os
import shutil
import argparse

from utils.train import train
from utils.data import get_data
from utils.test import test_model
from utils.validate import validate
from utils.utils import load_config, get_model, create_model_run_folder, save_model_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    opt = parser.parse_args()

    path_config_file = opt.path_config_file
    print('Path to config file:', path_config_file)
    config = load_config(path_config_file)

    model = get_model(config)
    model_name = config['model_name']

    run_path = create_model_run_folder(os.path.join('runs', model_name))
    shutil.copy2(path_config_file, os.path.join(run_path, os.path.basename(path_config_file)))
    save_model_plot(run_path, model_name, model)
    model_path = f'{run_path}/best_{model_name}.hdf5'

    train_data, valid_data, test_data = get_data(config)

    epochs = config['epochs']
    history = train(model, train_data, valid_data, epochs, model_path)

    val_num_batches = 1
    metrics = ['dice_score', 'true_positive_rate']
    validate(model, model_name, model_path, history, valid_data, val_num_batches, metrics, run_path)

    img_size = config['img_size']
    test_folder = f'{run_path}/tests'
    test_num_batches = 32 // config['batch_size']
    test_data_folder = os.path.join(config['destination_dataset_path'], 'test')
    test_model(model, model_path, test_data, test_num_batches, test_folder)
