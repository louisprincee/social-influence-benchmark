import datetime
import logging
import os
import numpy as np
import yaml
import pickle
from utils.arguments import Args


class DictAsObject:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


class MyTool:

    # =================================================================================================
    # Functions for this project
    # =================================================================================================


    @staticmethod
    def save_seeds(seeds, data_name, method, seed_num, diffusion_model):
        seed_path = os.path.join(MyTool.get_root_path(), 'save', 'seeds')
        if not os.path.exists(seed_path):
            os.makedirs(seed_path)
        seed_path = os.path.join(seed_path, data_name)
        if not os.path.exists(seed_path):
            os.makedirs(seed_path)
        seed_file = f"{method}_{diffusion_model}_{seed_num}.pkl"
        seed_path = os.path.join(seed_path, seed_file)
        with open(seed_path, 'wb') as f:
            pickle.dump(seeds, f)
            logging.info(f"saved seeds to {seed_path}")

    @staticmethod
    def load_seeds(seed_path):
        with open(seed_path, 'rb') as f:
            seeds = pickle.load(f)
            return seeds

    @staticmethod
    def get_root_path():
        return os.path.dirname(os.path.dirname(__file__))

    @staticmethod
    def get_data_path():
        return os.path.join(MyTool.get_root_path(), 'data')

    @staticmethod
    def get_method_special_data_path(method_name):
        target_path = os.path.join(MyTool.get_data_path(), 'method_run_data', f"{method_name}")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        return target_path


    @staticmethod
    def set_save_path(data_name, model_name, diffusion, seed_rate, save_root_name='save', TEST_MODE=False):
        root_path = MyTool.get_root_path()
        save_root_path = os.path.join(root_path, save_root_name)
        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
        save_path = os.path.join(save_root_path, data_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if TEST_MODE:
            folder_path = os.path.join(MyTool.get_root_path(), 'save', f'TEST_{model_name}')
        else:
            dt = datetime.datetime.now()
            folder_name = dt.strftime("%m%d_%H%M%S_") + model_name + "_" + diffusion + "_" + str(seed_rate)
            folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    @staticmethod
    def set_logging(save_path, model_name=None):
        """ set log file path and logging formats"""
        log_format = f"%(asctime)s | {model_name} | %(levelname)-4s | %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        logging.getLogger("").setLevel(logging.INFO)

        # save in log file
        filename = os.path.join(save_path, 'run.log')
        file = logging.FileHandler(filename)
        file.setFormatter(formatter)
        logging.getLogger("").addHandler(file)

    @staticmethod
    def set_easy_logging(model_name):
        log_format = f"%(asctime)s | {model_name} | %(levelname)-4s | %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        logging.getLogger("").setLevel(logging.INFO)

    @staticmethod
    def load_model_params(model_name, data_name):
        config_path = os.path.join(MyTool.get_root_path(), "configs", f"{model_name}.yaml")
        with open(config_path, 'r') as file:
            all_configs = yaml.safe_load(file)
            if data_name in all_configs.keys():
                all_configs = all_configs[data_name]
            else:
                if 'default' not in all_configs.keys():
                    raise ValueError(f"please set default config for {model_name}")
                logging.info(f"load default configs for {model_name}")
                all_configs = all_configs['default']
        train_configs = Args(**all_configs['train'])
        model_configs = Args(**all_configs['model'])
        return train_configs, model_configs

    @staticmethod
    def save_model_params(model_params, save_path):
        config_path = os.path.join(save_path, "config.yaml")
        with open(config_path, 'w') as file:
            yaml.dump(model_params, file)

    @staticmethod
    def save_to_yaml(args, cfgs, file_path):
        save_args = {
            'train': args._data,
            'model': cfgs._data,
        }
        # save as yaml file!
        with open(file_path, 'w') as file:
            yaml.dump(save_args, file)

    @staticmethod
    def get_diffusion_file_name(data_name, diffusion, mode, num):
        return f"{data_name}_{diffusion}_{mode}_{num}_cas.pkl"


if __name__ == '__main__':

    args, cfgs = MyTool.load_model_params('PIANO', 'cora_ml')

    save_path = os.path.join(MyTool.get_root_path(), 'save', 'test.yaml')
    MyTool.save_to_yaml(args, cfgs, save_path)

