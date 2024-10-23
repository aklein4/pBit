import torch

from transformers import AutoTokenizer

from models.hlm import HLmConfig, HLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'test-hlm'


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None, clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert len(tokenizer) == constants.GPT2_VOCAB_SIZE
    assert tokenizer.pad_token_id == constants.GPT2_PAD_TOKEN
    assert tokenizer.bos_token_id == constants.GPT2_BOS_TOKEN
    assert tokenizer.eos_token_id == constants.GPT2_EOS_TOKEN

    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids
    mask = torch.ones_like(x).bool() # torch.randint_like(x, 2).bool()
    mask[0, :3] = False
    mask[1, :8] = False

    print("loading model...")
    config = load_model_config(MODEL_CONFIG)
    model = HLmModel(HLmConfig(**config))

    print(f"Encoder: {sum([p.numel() for p in model.encoder.parameters()]):_}")
    print(f"Generator: {sum([p.numel() for p in model.generator.parameters()]):_}")
    print(f"Decoder: {sum([p.numel() for p in model.decoder.parameters()]):_}")

    logits, kl, smooth_kl = model(x, mask)

    print(logits.shape)
    print(kl)
    print(smooth_kl)


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
