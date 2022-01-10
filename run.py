# argument
import argparse

from pprgo.run_function import run_experiment

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PPR-Go')
    parser.add_argument('--config', nargs="+", type=str, default='config/normal_cora_full_demo.yaml',
                        help='Path to the config file')
    config_path_list = parser.parse_args().config
    for config_path in config_path_list:
        run_experiment(config_path)
