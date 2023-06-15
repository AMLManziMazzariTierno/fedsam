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
    'cifar10.cnn': (0.01, 10), #lr, #classes
    'cifar100.resnet20': (0.01, 100)
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

# Fed-CCVR configuration
conf = {

	# Data type: tabular, image
	"data_type" : "image",

	# Model selection: mlp, simple-cnn, vgg, resnet20
	"model_name" : "resnet20",

	# Processing method: fed_ccvr
	"no-iid": "fed_ccvr",

	# Global epochs
	"global_epochs" : 1000,

	# Local epochs
	"local_epochs" : 3,

	# Dirichlet parameter
	"beta" : 1000,

	"batch_size" : 128,

	"weight_decay": 0.0001,

    # Learning rate
	"lr" : 0.1,

	"momentum" : 0.9,

	# Number of classes
	"num_classes": 100, # era a 2

	# Number of parties/nodes
	"num_parties": 8,

    # Model aggregation weight initialization
	"is_init_avg": True,

    # Local validation set split ratio
	"split_ratio": 0.3,

    # Label column name
	"label_column": "label",

	# Data column name
	"data_column": "file",

    # Test data
	"test_dataset": "./data/cifar10/test/test.csv",

    # Training data
	"train_dataset" : "./data/cifar10/train/train.csv",

    # Model save directory
	"model_dir":"./save_model/",

    # Model filename
	"model_file":"model.pth",

	"retrain":{
		"epoch": 50,
		"lr": 0.001,
		"num_vr":2000
	}
}