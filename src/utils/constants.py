
import os

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    print("Warning: torch_xla not found", flush=True)
    XLA_AVAILABLE = False

# get the base path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)

# device of current process
XLA_DEVICE = lambda: xm.xla_device()

# local id of the current device
XLA_LOCAL_RANK = lambda: xm.get_local_ordinal()

# id of the current device
XLA_RANK = lambda: xm.get_ordinal()

# whether this is the main process on its device
XLA_LOCAL_MAIN = lambda: xm.is_master_ordinal(local=True)

# whether this is the main process
XLA_MAIN = lambda: xm.is_master_ordinal(local=False)

# number of devices
NUM_XLA_DEVICES = lambda: xm.xrt_world_size()

# local data path
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# paths to config files
MODEL_CONFIG_PATH = os.path.join(BASE_PATH, "model_configs")
TRAIN_CONFIG_PATH = os.path.join(BASE_PATH, "train_configs")

# gpt2 tokenizer
GPT2_TOKENIZER = 'openai-community/gpt2'

# gpt2 tokenizer vocab size (with pad)
GPT2_VOCAB_SIZE = 50258

# gpt2 tokenizer built in special tokens
GPT2_BOS_TOKEN = 50256
GPT2_EOS_TOKEN = 50256

# gpt2 tokenizer pad token id after adding [PAD]
GPT2_PAD_TOKEN = 50257

# huggingface login id
HF_ID = "aklein4"
