import torch
import math

class cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 128
    test_batch_size = 128

    input_size = 224
    epochs = 1             # number of epochs
    lr = 0.01               # do not use bigger than 0.01
    momentum = 0.9
    seed = 1
    lr_step_size = 2        # step size for learning rate decay
    lr_gamma = 0.5          # learning rate decay
    lasso_lambda = 1e-5     # weight regularization

    # QAT cfg
    activation_QAT_start = 0
    num_bits = 8
    symmetrical = True
    stats_mode = "entropy"   # minmax or entropy
    activation_bit = None   # remember to turn this off if you do not need it . (None is off)

    # pruning
    pruning = False
    pruning_every = 2
    pruning_ratio = 0.1     # prune a fixed pruning ratio
    final_sparsity = 50
    prune_after_epoch = 0   # start pruning after this epoch


    # log config
    log_interval = 40
    save_model = True
    no_cuda = True

    # file path cfg
    dataset_root = "./data"
    logger_path = './logger/log'
    models_path = './models'
