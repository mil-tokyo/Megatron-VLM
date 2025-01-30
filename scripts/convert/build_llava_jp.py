import argparse
import os
import shutil
import time
import warnings

import torch
from modeling_llava import LlavaForConditionalGeneration
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPVisionModel,
    LlavaConfig,
    SiglipVisionModel,
)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--save-dir', type=str,
        help='Save directory'
    )
    parser.add_argument(
        '--text-model', type=str,
        help='Text model name on huggingface hub; Currently, only LLaMA2 based architectures are supported',
        default="llm-jp/llm-jp-3-13b-instruct"
    )
    parser.add_argument(
        '--vision-model', type=str,
        help='Vision model name on huggingface hub; '
             'tested: openai/clip-vit-large-patch14-336, google/siglip-so400m-patch14-384',
        default="google/siglip-so400m-patch14-384"
    )
    parser.add_argument('--vocab-size-padding', action='store_true', help='Pad vocab size to be divisible by 128')
    parser.add_argument('--expand-projector', action='store_true', help='Expand projector hidden size')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    original_model_name = "llava-hf/llava-1.5-7b-hf"
    text_model_name = args.text_model
    vision_model_name = args.vision_model

    config = LlavaConfig.from_pretrained(original_model_name)
    text_config = AutoConfig.from_pretrained(text_model_name)
    vision_config = AutoConfig.from_pretrained(vision_model_name).vision_config

    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>', '<pad>']})
    tokenizer.pad_token_id = tokenizer.vocab['<pad>']

    vocab_size = len(tokenizer)
    if args.vocab_size_padding:
        # adjust vocab size (add <image> and <pad> tokens)
        print(f"original vocab size: {vocab_size}")
        # set vocab size to be divisible by 128
        vocab_size = vocab_size + (128 - vocab_size % 128)
        text_config.vocab_size = vocab_size
        print(f" -> new vocab size: {vocab_size}")
    else:
        text_config.vocab_size = vocab_size
        print(f"vocab size: {vocab_size}")

    # replace text config
    config.text_config = text_config
    config.vocab_size = vocab_size

    # update model config
    config.image_token_index = tokenizer.vocab['<image>']
    config.pad_token_id = tokenizer.vocab['<pad>']

    # update vision config
    if "siglip" in vision_model_name:
        projection_dim = config.vision_config.projection_dim
        config.vision_config = vision_config
        config.vision_config.projection_dim = projection_dim
    config.vision_config.vocab_size = vocab_size

    # update projector config
    if args.expand_projector:
        config.projector_hidden_size = config.text_config.hidden_size * 2
    else:
        config.projector_hidden_size = config.text_config.hidden_size

    # define model
    print("Building model...")
    start = time.time()
    model = LlavaForConditionalGeneration(config)
    elapsed = time.time() - start
    if elapsed > 60:
        print(f" └ model built in {elapsed / 60:.2f} min {elapsed % 60:.2f} sec")
    else:
        print(f" └ model built in {elapsed:.2f} sec")
    print(f"model: \n{model}")

    # load vision model & text model
    print(f"\nLoading vision model ({vision_model_name})...")
    if "siglip" in vision_model_name:
        vision_model = SiglipVisionModel.from_pretrained(vision_model_name)
    else:
        vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
    print(" └ vision model loaded!")

    print("\nLoading text model...")
    text_model = AutoModelForCausalLM.from_pretrained(text_model_name)
    text_model.resize_token_embeddings(vocab_size)
    # check if out_features size of q_proj, k_proj, v_proj is equal to each other
    q_proj_size = text_model.model.layers[0].self_attn.q_proj.weight.size(0)
    k_proj_size = text_model.model.layers[0].self_attn.k_proj.weight.size(0)
    v_proj_size = text_model.model.layers[0].self_attn.v_proj.weight.size(0)
    if q_proj_size != k_proj_size or q_proj_size != v_proj_size:
        warnings.warn(f"q_proj, k_proj, and v_proj should have the same size, "
                      f"but got q_proj: {q_proj_size}, k_proj: {k_proj_size}, v_proj: {v_proj_size}")
    print(" └ text model loaded!")

    # set vision model weight
    print("\nSetting vision model weight...")
    model.vision_tower.vision_model.load_state_dict(vision_model.vision_model.state_dict())
    print(" └ vision model weight set!")
    print("\nSetting text model weight...")
    model.language_model.model.load_state_dict(text_model.model.state_dict())
    print(" └ text model weight set!")

    print("\nconvert model to bfloat16")
    model = model.to(torch.bfloat16)
    print(" └ model converted to bfloat16!")

    print("\nsaving model...")
    model.save_pretrained(args.save_dir)
    print(f" └ Model saved to {args.save_dir}")

    print("\nSaving text config...")
    text_config_dir = f"{args.save_dir}/text_config"
    os.makedirs(text_config_dir, exist_ok=True)
    text_config.save_pretrained(text_config_dir)
    # move text config to save_dir and rename it to text_config.json
    shutil.move(f"{text_config_dir}/config.json", f"{args.save_dir}/text_config.json")
    # remove text_config directory
    shutil.rmtree(text_config_dir)
    print(f" └ Text config saved to {args.save_dir}")

    # save processor
    print("\nSaving processor...")
    processor = AutoProcessor.from_pretrained(original_model_name)
    processor.save_pretrained(args.save_dir)
    print(f" └ Processor saved to {args.save_dir}")

    # save tokenizer
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained(args.save_dir)
    print(f" └ Tokenizer saved to {args.save_dir}")

    # save preprocessor
    print("\nSaving preprocessor...")
    if "siglip" in vision_model_name:
        preprocessor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    else:
        preprocessor = AutoProcessor.from_pretrained(original_model_name)
    # overwrite preprocessor's tokenizer
    preprocessor.tokenizer = tokenizer
    preprocessor.save_pretrained(args.save_dir)
    print(f" └ Preprocessor saved to {args.save_dir}")

    # calculate model size
    model_size = sum(p.numel() for p in model.parameters())
    vision_model_size = sum(p.numel() for p in model.vision_tower.vision_model.parameters())
    text_model_size = sum(p.numel() for p in model.language_model.parameters())
    projector_size = sum(p.numel() for p in model.multi_modal_projector.parameters())
    print(f"\nModel size: {model_size:,}, vision model size: {vision_model_size:,}, "
          f"text model size: {text_model_size:,}, projector size: {projector_size:,}")

    # save model printout
    with open(os.path.join(args.save_dir, "model.txt"), "w") as f:
        f.write(str(model))


if __name__ == '__main__':
    main()
