"""Constants"""
DEVICE = 'cuda' # 'cuda' or 'cpu'
NUM_CLASSES = 100
BATCH_SIZE = 128
LR = 0.1            # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 0.0001  # Regularization, you can keep this at the default
NUM_EPOCHS = 160      # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20       # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 1          # Multiplicative factor for learning rate step-down
LOG_FREQUENCY = 25   #frequenza con cui viene stampato qualcosa
PRETRAINED = True       # If 'True' the NETWORK will be pre-trained on ImageNet dataset
FREEZE = 'no_freezing'   # Available choice: 'no_freezing', 'conv_layers', 'fc_layers'
RANDOM = 42
# Data augmentation
AUG_PROB = 0.5   # the probability with witch each image is transformed at training time during each epoch

"""Configuration file for common models/experiments"""

MAIN_PARAMS = {
    'cifar100': {
        'small': (1000, 100, 10),
        'medium': (10000, 100, 10),
        'large': (20000, 100, 10)
        },
    'cifar10': {
        'small': (1000, 100, 10),
        'medium': (10000, 100, 10),
        'large': (20000, 100, 10)
        },
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'cifar100.cnn': (0.01, 100),
    'cifar10.cnn': (0.01, 10),
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
CLIENT_PARAMS_KEY = 'client_params_norm'
CLIENT_GRAD_KEY = 'client_grad_norm'
CLIENT_TASK_KEY = 'client_task'
