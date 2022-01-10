import os
import time
import logging
import yaml
import ast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

from .utils import get_data, get_max_memory_bytes
from .ppr import topk_ppr_matrix
from .pprgo import PPRGo, SAPPRGo
from .train import train, train_sappr
from .predict import predict, predict_sappr
from .dataset import PPRDataset, SAPPRDataset

# Set up logging
from datetime import datetime
import logging
from logging import StreamHandler, FileHandler, Formatter
from logging import INFO, DEBUG, NOTSET


def run_one_pprgo_experiment(config):
    logger = logging.getLogger(__name__)
    logger.info('run_one_pprgo_experiment')
    '''
    # Set config
    '''
    data_file = config['data_file']           # Path to the .npz data file
    split_seed = config['seed']
    # split_seed = config['split_seed']          # Seed for splitting the dataset into train/val/test
    ntrain_div_classes = config['ntrain_div_classes']  # Number of training nodes divided by number of classes
    attr_normalization = config['attr_normalization']  # Attribute normalization. Not used in the paper

    alpha = config['alpha']               # PPR teleport probability
    eps = config['eps']                 # Stopping threshold for ACL's ApproximatePR
    topk = config['topk']                # Number of PPR neighbors for each node
    ppr_normalization = config['ppr_normalization']   # Adjacency matrix normalization for weighting neighbors

    hidden_size = config['hidden_size']         # Size of the MLP's hidden layer
    nlayers = config['nlayers']             # Number of MLP layers
    weight_decay = config['weight_decay']        # Weight decay used for training the MLP
    dropout = config['dropout']             # Dropout used for training

    lr = config['lr']                  # Learning rate
    max_epochs = config['max_epochs']          # Maximum number of epochs (exact number if no early stopping)
    batch_size = config['batch_size']          # Batch size for training
    batch_mult_val = config['batch_mult_val']      # Multiplier for validation batch size

    eval_step = config['eval_step']           # Accuracy is evaluated after every this number of steps
    run_val = config['run_val']             # Evaluate accuracy on validation set during training

    early_stop = config['early_stop']          # Use early stopping
    patience = config['patience']            # Patience for early stopping

    nprop_inference = config['nprop_inference']     # Number of propagation steps during inference
    inf_fraction = config['inf_fraction']        # Fraction of nodes for which local predictions are computed during inference

    '''
    # Load data
    '''
    start = time.time()
    (adj_matrix, attr_matrix, labels,
     train_idx, val_idx, test_idx) = get_data(
        f"{data_file}",
        seed=split_seed,
        ntrain_div_classes=ntrain_div_classes,
        normalize_attr=attr_normalization
    )
    try:
        d = attr_matrix.n_columns
    except AttributeError:
        d = attr_matrix.shape[1]
    nc = labels.max() + 1
    time_loading = time.time() - start
    logger.info(f"Load data Runtime: {time_loading:.2f}s")

    # Graph Information
    logger.info(f'adj_matrix:{adj_matrix.shape}')
    logger.info(f'attr_matrix:{attr_matrix.shape}')
    logger.info(f'labels:{len(np.unique(labels))}')
    logger.info(f'train_idx: {ntrain_div_classes}*{len(np.unique(labels))}={train_idx.shape}')
    logger.info(f'val_idx: {val_idx.shape}')
    logger.info(f'test_idx:{test_idx.shape}')

    '''
    # Preprocessing: Calculate PPR scores
    '''
    # compute the ppr vectors for train/val nodes using ACL's ApproximatePR
    start = time.time()
    topk_train = topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk,
                                 normalization=ppr_normalization)
    train_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_train, indices=train_idx, labels_all=labels)
    if run_val:
        topk_val = topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk,
                                   normalization=ppr_normalization)
        val_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_val, indices=val_idx, labels_all=labels)
    else:
        val_set = None
    time_preprocessing = time.time() - start
    logger.info(f"Preprocessing Runtime: {time_preprocessing:.2f}s")

    '''
    # Training: Set up model and train
    '''
    start = time.time()
    model = PPRGo(d, nc, hidden_size, nlayers, dropout)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    nepochs, _, _ = train(
        model=model, train_set=train_set, val_set=val_set,
        lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, batch_size=batch_size, batch_mult_val=batch_mult_val,
        eval_step=eval_step, early_stop=early_stop, patience=patience)
    time_training = time.time() - start
    logging.info(f"Training Runtime: {time_training:.2f}s")

    '''
    # Inference (val and test)
    '''
    start = time.time()
    predictions, time_logits, time_propagation = predict(
        model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
        nprop=nprop_inference, inf_fraction=inf_fraction,
        ppr_normalization=ppr_normalization)
    time_inference = time.time() - start
    logger.info(f"Runtime: {time_inference:.2f}s")

    '''
    # Collect and print results
    '''
    acc_train = 100 * accuracy_score(labels[train_idx], predictions[train_idx])
    acc_val = 100 * accuracy_score(labels[val_idx], predictions[val_idx])
    acc_test = 100 * accuracy_score(labels[test_idx], predictions[test_idx])
    f1_train = f1_score(labels[train_idx], predictions[train_idx], average='macro')
    f1_val = f1_score(labels[val_idx], predictions[val_idx], average='macro')
    f1_test = f1_score(labels[test_idx], predictions[test_idx], average='macro')

    gpu_memory = torch.cuda.max_memory_allocated()
    memory = get_max_memory_bytes()

    time_total = time_preprocessing + time_training + time_inference
    logger.info(f'''
        Accuracy: Train: {acc_train:.1f}%, val: {acc_val:.1f}%, test: {acc_test:.1f}%
        F1 score: Train: {f1_train:.3f}, val: {f1_val:.3f}, test: {f1_test:.3f}

        Runtime: Preprocessing: {time_preprocessing:.2f}s, training: {time_training:.2f}s, inference: {time_inference:.2f}s -> total: {time_total:.2f}s
        Memory: Main: {memory / 2**30:.2f}GB, GPU: {gpu_memory / 2**30:.3f}GB
        ''')

    return acc_train, acc_val, acc_test, f1_train, f1_val, f1_test, time_total, time_preprocessing, time_training, time_inference, gpu_memory, memory


def run_one_sapprgo_experiment(config):
    '''
    semi-adaptive pprgo
    '''

    logger = logging.getLogger(__name__)
    logger.info('run_one_sapprgo_experiment')

    '''
    # Set config
    '''
    data_file = config['data_file']           # Path to the .npz data file
    split_seed = config['seed']
    # split_seed = config['split_seed']          # Seed for splitting the dataset into train/val/test
    ntrain_div_classes = config['ntrain_div_classes']  # Number of training nodes divided by number of classes
    attr_normalization = config['attr_normalization']  # Attribute normalization. Not used in the paper

    nsplit_alpha = config['nsplit_alpha']  # Number of splits for alpha
    logspace = config['logspace']  # Whether to use logspace for alpha
    min_alpha = config['min_alpha']        # Minimum alpha
    max_alpha = config['max_alpha']        # Maximum alpha
    # alpha = config['alpha']               # PPR teleport probability
    eps = config['eps']                 # Stopping threshold for ACL's ApproximatePR
    topk = config['topk']                # Number of PPR neighbors for each node
    ppr_normalization = config['ppr_normalization']   # Adjacency matrix normalization for weighting neighbors

    hidden_size = config['hidden_size']         # Size of the MLP's hidden layer
    nlayers = config['nlayers']             # Number of MLP layers
    weight_decay = config['weight_decay']        # Weight decay used for training the MLP
    dropout = config['dropout']             # Dropout used for training

    lr = config['lr']                  # Learning rate
    max_epochs = config['max_epochs']          # Maximum number of epochs (exact number if no early stopping)
    batch_size = config['batch_size']          # Batch size for training
    batch_mult_val = config['batch_mult_val']      # Multiplier for validation batch size

    eval_step = config['eval_step']           # Accuracy is evaluated after every this number of steps
    run_val = config['run_val']             # Evaluate accuracy on validation set during training

    early_stop = config['early_stop']          # Use early stopping
    patience = config['patience']            # Patience for early stopping

    nprop_inference = config['nprop_inference']     # Number of propagation steps during inference
    inf_fraction = config['inf_fraction']        # Fraction of nodes for which local predictions are computed during inference

    if logspace:
        alphas = np.logspace(np.log10(min_alpha), np.log10(max_alpha), nsplit_alpha, base=10)
    else:
        alphas = np.linspace(min_alpha, max_alpha, nsplit_alpha)
    logger.info('alphas: {}'.format(alphas))

    '''
    # Load data
    '''
    start = time.time()
    (adj_matrix, attr_matrix, labels,
     train_idx, val_idx, test_idx) = get_data(
        f"{data_file}",
        seed=split_seed,
        ntrain_div_classes=ntrain_div_classes,
        normalize_attr=attr_normalization
    )
    try:
        d = attr_matrix.n_columns
    except AttributeError:
        d = attr_matrix.shape[1]
    nc = labels.max() + 1
    time_loading = time.time() - start
    logger.info(f"Load data Runtime: {time_loading:.2f}s")
    # Graph Information
    logger.info(f'adj_matrix:{adj_matrix.shape}')
    logger.info(f'attr_matrix:{attr_matrix.shape}')
    logger.info(f'labels:{len(np.unique(labels))}')
    logger.info(f'train_idx: {ntrain_div_classes}*{len(np.unique(labels))}={train_idx.shape}')
    logger.info(f'val_idx: {val_idx.shape}')
    logger.info(f'test_idx:{test_idx.shape}')

    '''
    # Preprocessing: Calculate PPR scores
    '''
    # compute the ppr vectors for train/val nodes using ACL's ApproximatePR
    start = time.time()

    topk_train_list = []
    for alpha in alphas:
        topk_train = topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk,
                                     normalization=ppr_normalization)
        topk_train_list.append(topk_train)
    train_set = SAPPRDataset(attr_matrix_all=attr_matrix, ppr_matrix_list=topk_train_list, indices=train_idx, labels_all=labels)

    if run_val:
        topk_val_list = []
        for alpha in alphas:
            topk_val = topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk,
                                       normalization=ppr_normalization)
            topk_val_list.append(topk_val)
        val_set = SAPPRDataset(attr_matrix_all=attr_matrix, ppr_matrix_list=topk_val_list, indices=val_idx, labels_all=labels)
    else:
        val_set = None

    time_preprocessing = time.time() - start
    logger.info(f"Preprocessing Runtime: {time_preprocessing:.2f}s")

    '''
    # Training: Set up model and train
    '''
    start = time.time()
    model = SAPPRGo(d, nc, nsplit_alpha, hidden_size, nlayers, dropout)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    nepochs, _, _ = train_sappr(
        model=model, train_set=train_set, val_set=val_set,
        lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, batch_size=batch_size, batch_mult_val=batch_mult_val,
        eval_step=eval_step, early_stop=early_stop, patience=patience)
    time_training = time.time() - start
    logging.info(f"Training Runtime: {time_training:.2f}s")

    '''
    # Inference (val and test)
    '''
    start = time.time()
    predictions, time_logits, time_propagation = predict_sappr(
        model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alphas=alphas,
        nprop=nprop_inference, inf_fraction=inf_fraction,
        ppr_normalization=ppr_normalization)
    time_inference = time.time() - start
    logger.info(f"Runtime: {time_inference:.2f}s")

    '''
    # Collect and print results
    '''
    acc_train = 100 * accuracy_score(labels[train_idx], predictions[train_idx])
    acc_val = 100 * accuracy_score(labels[val_idx], predictions[val_idx])
    acc_test = 100 * accuracy_score(labels[test_idx], predictions[test_idx])
    f1_train = f1_score(labels[train_idx], predictions[train_idx], average='macro')
    f1_val = f1_score(labels[val_idx], predictions[val_idx], average='macro')
    f1_test = f1_score(labels[test_idx], predictions[test_idx], average='macro')

    gpu_memory = torch.cuda.max_memory_allocated()
    memory = get_max_memory_bytes()

    time_total = time_preprocessing + time_training + time_inference
    logger.info(f'''
        Accuracy: Train: {acc_train:.1f}%, val: {acc_val:.1f}%, test: {acc_test:.1f}%
        F1 score: Train: {f1_train:.3f}, val: {f1_val:.3f}, test: {f1_test:.3f}

        Runtime: Preprocessing: {time_preprocessing:.2f}s, training: {time_training:.2f}s, inference: {time_inference:.2f}s -> total: {time_total:.2f}s
        Memory: Main: {memory / 2**30:.2f}GB, GPU: {gpu_memory / 2**30:.3f}GB
        ''')

    return acc_train, acc_val, acc_test, f1_train, f1_val, f1_test, time_total, time_preprocessing, time_training, time_inference, gpu_memory, memory


def run_one_experiment(config):
    # config に nsplit_alpha が含まれていれば run_one_sapprgo_experiment として実行
    if 'nsplit_alpha' in config:
        return run_one_sapprgo_experiment(config)
    else:
        return run_one_pprgo_experiment(config)


def run_experiment(config_path):
    config_name = os.path.basename(config_path)
    config_name = os.path.splitext(config_name)[0]

    '''
    # Set up logging
    '''
    formatter = Formatter(
        fmt='%(asctime)s (%(levelname)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    # ストリームハンドラの設定
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)
    # 保存先の有無チェック
    if not os.path.isdir('./log'):
        os.makedirs('./log', exist_ok=True)
    # ファイルハンドラの設定
    file_handler = FileHandler(
        f"./log/{config_name}.log"  # _{datetime.now():%Y%m%d%H%M%S}
    )
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)
    # ルートロガーの設定
    logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])
    logger = logging.getLogger(__name__)

    logger.info('############################################################')
    '''
    # Load config
    '''
    with open(config_path, 'r') as c:
        config = yaml.safe_load(c)
    logger.info(f'config: {config}')

    # For strings that yaml doesn't parse (e.g. None)
    for key, val in config.items():
        if isinstance(val, str):
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass

    acc_train_list = []
    acc_val_list = []
    acc_test_list = []
    f1_train_list = []
    f1_val_list = []
    f1_test_list = []
    time_total_list = []
    time_preprocessing_list = []
    time_training_list = []
    time_inference_list = []
    gpu_memory_list = []
    memory_list = []

    for exid in range(config['nexp']):
        logger.info(f'Experiment {exid}')
        # Set seed
        config['seed'] += 1  # シード値を変える
        seed = config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Run experiment
        acc_train, acc_val, acc_test, f1_train, f1_val, f1_test, time_total, time_preprocessing, time_training, time_inference, gpu_memory, memory \
            = run_one_experiment(config)
        acc_train_list.append(acc_train)
        acc_val_list.append(acc_val)
        acc_test_list.append(acc_test)
        f1_train_list.append(f1_train)
        f1_val_list.append(f1_val)
        f1_test_list.append(f1_test)
        time_total_list.append(time_total)
        time_preprocessing_list.append(time_preprocessing)
        time_training_list.append(time_training)
        time_inference_list.append(time_inference)
        gpu_memory_list.append(gpu_memory)
        memory_list.append(memory)

    # Print results (mean and std)
    logger.info('Results: mean and std')
    logger.info(f'''
        Accuracy: Train: {np.mean(acc_train_list):.1f}({np.std(acc_train_list):.1f})%, val: {np.mean(acc_val_list):.1f}({np.std(acc_val_list):.1f})%, test: {np.mean(acc_test_list):.1f}({np.std(acc_test_list):.1f})%
        F1 score: Train: {np.mean(f1_train_list):.3f}({np.std(f1_train_list):.3f}), val: {np.mean(f1_val_list):.3f}({np.std(f1_val_list):.3f}), test: {np.mean(f1_test_list):.3f}({np.std(f1_test_list):.3f})

        Runtime: Preprocessing: {np.mean(time_preprocessing_list):.2f}({np.std(time_preprocessing_list):.2f})s, training: {np.mean(time_training_list):.2f}({np.std(time_training_list):.2f})s, inference: {np.mean(time_inference_list):.2f}({np.std(time_inference_list):.2f})s -> total: {np.mean(time_total_list):.2f}({np.std(time_total_list):.2f})s
        Memory: Main: {np.mean(memory_list) / 2**30:.2f}({np.std(memory_list) / 2**30:.2f})GB, GPU: {np.mean(gpu_memory_list) / 2**30:.3f}({np.std(gpu_memory_list) / 2**30:.3f})GB
        ''')

    # Save results
    with open('log/result.txt', mode='a') as f:
        row_ltsv = ''
        for key, val in config.items():
            row_ltsv += f'{key}: {val}\t'
        row_ltsv += f'acc_train_mean: {np.mean(acc_train_list):.1f}\t'
        row_ltsv += f'acc_train_std: {np.std(acc_train_list):.1f}\t'
        row_ltsv += f'acc_val_mean: {np.mean(acc_val_list):.1f}\t'
        row_ltsv += f'acc_val_std: {np.std(acc_val_list):.1f}\t'
        row_ltsv += f'acc_test_mean: {np.mean(acc_test_list):.1f}\t'
        row_ltsv += f'acc_test_std: {np.std(acc_test_list):.1f}\t'
        row_ltsv += f'f1_train_mean: {np.mean(f1_train_list):.3f}\t'
        row_ltsv += f'f1_train_std: {np.std(f1_train_list):.3f}\t'
        row_ltsv += f'f1_val_mean: {np.mean(f1_val_list):.3f}\t'
        row_ltsv += f'f1_val_std: {np.std(f1_val_list):.3f}\t'
        row_ltsv += f'f1_test_mean: {np.mean(f1_test_list):.3f}\t'
        row_ltsv += f'f1_test_std: {np.std(f1_test_list):.3f}\t'
        row_ltsv += f'time_preprocessing_mean: {np.mean(time_preprocessing_list):.2f}\t'
        row_ltsv += f'time_preprocessing_std: {np.std(time_preprocessing_list):.2f}\t'
        row_ltsv += f'time_training_mean: {np.mean(time_training_list):.2f}\t'
        row_ltsv += f'time_training_std: {np.std(time_training_list):.2f}\t'
        row_ltsv += f'time_inference_mean: {np.mean(time_inference_list):.2f}\t'
        row_ltsv += f'time_inference_std: {np.std(time_inference_list):.2f}\t'
        row_ltsv += f'time_total_mean: {np.mean(time_total_list):.2f}\t'
        row_ltsv += f'time_total_std: {np.std(time_total_list):.2f}\t'
        row_ltsv += f'memory_mean: {np.mean(memory_list) / 2**30:.2f}\t'
        row_ltsv += f'memory_std: {np.std(memory_list) / 2**30:.2f}\t'
        row_ltsv += f'gpu_memory_mean: {np.mean(gpu_memory_list) / 2**30:.3f}\t'
        row_ltsv += f'gpu_memory_std: {np.std(gpu_memory_list) / 2**30:.3f}\t'
        f.write(row_ltsv + '\n')
        print(row_ltsv)
