import torch
import math

class cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    test_batch_size = 128

    input_size = 224
    epochs = 3
    lr = 0.01
    momentum = 0.9
    seed = 1
    lr_step_size = 10
    lr_gamma = 0.1 

    # QAT cfg
    start_QAT_epoch = math.floor(epochs/2)
    num_bits = 8
    symmetrical = True

    # pruning
    pruning = True
    pruning_every = 2
    pruning_ratio = 0.1
    final_sparsity = 75

    # log config
    log_interval = 40
    save_model = True
    no_cuda = True

    # file path cfg
    dataset_root = "./data"
    logger_path = './logger/log'
    models_path = './models'
