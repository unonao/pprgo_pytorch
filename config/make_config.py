import yaml
import copy
import numpy as np


def float_representer(dumper, value):
    text = '{0:.4f}'.format(value)
    return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)


yaml.add_representer(float, float_representer)

data_file_list = ['data/pubmed.npz', 'data/cora_full.npz', 'data/reddit.npz']
min_alpha = 1e-3
max_alpha = 9e-1
# for sapprgo
nsplit_alpha_list = [i for i in range(1, 11)]
base_dct = yaml.safe_load('''
data_file: data/pubmed.npz  # Path to the .npz data file
nexp: 5                     # Number of experiments
seed: 0
#split_seed: 0               # Seed for splitting the dataset into train/val/test
ntrain_div_classes: 20      # Number of training nodes divided by number of classes
attr_normalization: None    # Attribute normalization. Not used in the paper
eps: 1e-4                   # Stopping threshold for ACL's ApproximatePR
topk: 32                    # Number of PPR neighbors for each node
ppr_normalization: 'sym'    # Adjacency matrix normalization for weighting neighbors
hidden_size: 32             # Size of the MLP's hidden layer
nlayers: 2                  # Number of MLP layers
weight_decay: 1e-4          # Weight decay used for training the MLP
dropout: 0.1                # Dropout used for training
lr: 5e-3                    # Learning rate
max_epochs: 200             # Maximum number of epochs (exact number if no early stopping)
batch_size: 512             # Batch size for training
batch_mult_val: 4           # Multiplier for validation batch size
eval_step: 10               # Accuracy is evaluated after every this number of steps
run_val: True              # Evaluate accuracy on validation set during training
early_stop: True           # Use early stopping
patience: 50                # Patience for early stopping
nprop_inference: 2          # Number of propagation steps during inference
inf_fraction: 1.0           # Fraction of nodes for which local predictions are computed during inference
''')
base_dct['min_alpha'] = min_alpha
base_dct['max_alpha'] = max_alpha

file_list = []

# sapprgo
for data_file in data_file_list:
    for nsplit_alpha in nsplit_alpha_list:
        # logspace is True
        dct = copy.deepcopy(base_dct)
        dct['data_file'] = data_file
        dct['nsplit_alpha'] = nsplit_alpha
        dct['logspace'] = True
        config_file = 'config/sapprgo_' + data_file.split('/')[-1].split('.')[0] + '_logspace_nsplit' + str(nsplit_alpha).zfill(2) + '.yaml'
        file_list.append(config_file)
        with open(config_file, 'w') as f:
            yaml.dump(dct, f, default_flow_style=False)
        # logspace is False
        dct = copy.deepcopy(base_dct)
        dct['data_file'] = data_file
        dct['nsplit_alpha'] = nsplit_alpha
        dct['logspace'] = False
        config_file = 'config/sapprgo_' + data_file.split('/')[-1].split('.')[0] + '_linspace_nsplit' + str(nsplit_alpha).zfill(2) + '.yaml'
        file_list.append(config_file)
        with open(config_file, 'w') as f:
            yaml.dump(dct, f, default_flow_style=False)

# normal
for data_file in data_file_list:
    # 1e-3 から 9e-1 まで logspaceで10個のalphaを使う
    for alpha in np.logspace(np.log10(min_alpha), np.log10(max_alpha), 10):
        dct = copy.deepcopy(base_dct)
        dct['data_file'] = data_file
        dct['alpha'] = float(alpha)
        config_file = 'config/normal_' + data_file.split('/')[-1].split('.')[0] + f'_alpha{alpha:.4f}' + '.yaml'
        file_list.append(config_file)
        with open(config_file, 'w') as f:
            yaml.dump(dct, f, default_flow_style=False)

print(f'python run.py --config ' + ' '.join(file_list))
