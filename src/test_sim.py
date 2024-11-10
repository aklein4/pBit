import torch

from transformers import AutoTokenizer

from models.base import BaseConfig
from models.sim import SimLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'mini-base'


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None, clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert len(tokenizer) == constants.GPT2_VOCAB_SIZE
    assert tokenizer.pad_token_id == constants.GPT2_PAD_TOKEN
    assert tokenizer.bos_token_id == constants.GPT2_BOS_TOKEN
    assert tokenizer.eos_token_id == constants.GPT2_EOS_TOKEN

    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids

    print("loading model...")
    config = load_model_config(MODEL_CONFIG)
    model = SimLmModel(BaseConfig(**config))

    out = model(x)

    # print(out)
    print(out.shape)

    print(model.get_sim_score())
    print(model.get_sim_score())


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
