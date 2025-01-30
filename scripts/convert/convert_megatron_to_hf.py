import argparse
import json
import os
import shutil
import subprocess


def replace_path(original_path: str, bind_mapping: list):
    for bind in bind_mapping:
        original_path = original_path.replace(bind.split(":")[0], bind.split(":")[1])
    return original_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--megatron-dir', type=str,
        help='Megatron checkpoint directory'
    )
    parser.add_argument(
        '--tokenizer-dir', type=str,
        help='Tokenizer model directory'
    )
    parser.add_argument(
        '--hf-dir', type=str,
        help='Huggingface checkpoint directory'
    )
    parser.add_argument('--tp', type=int, default=1, help='Number of tensor model parallel')
    parser.add_argument('--pp', type=int, default=1, help='Number of pipeline model parallel')
    parser.add_argument(
        '--saver', type=str, default="llava_hf",
        help='Saver name; e.g., llava_hf, llava3_hf, llava3_hf_siglip'
    )
    parser.add_argument(
        '--loader', type=str, default="mcore_llava",
        help='Loader name; e.g., mcore_llava, mcore_llava_siglip'
    )
    parser.add_argument(
        '--sif-path', type=str,
    )
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    # Configuration variables
    hf_checkpoint_dir = args.hf_dir
    os.makedirs(hf_checkpoint_dir, exist_ok=True)

    print(f"HF_CHECKPOINT_DIR: {hf_checkpoint_dir}")
    print(f"MEGATRON_CHECKPOINT_DIR: {args.megatron_dir}")
    print(f"TOKENIZER_MODEL: {args.tokenizer_dir}")

    # check config.json
    # copy from tokenizer dir
    shutil.copyfile(os.path.join(args.tokenizer_dir, "config.json"), os.path.join(hf_checkpoint_dir, "config.json"))
    print(f"copied config.json from {args.tokenizer_dir} -> {hf_checkpoint_dir}")
    # copy text_config.json
    shutil.copyfile(
        os.path.join(args.tokenizer_dir, "text_config.json"),
        os.path.join(hf_checkpoint_dir, "text_config.json")
    )
    print(f"copied text_config.json from {args.tokenizer_dir} -> {hf_checkpoint_dir}")
    # copy preprocessor_config.json
    shutil.copyfile(
        os.path.join(args.tokenizer_dir, "preprocessor_config.json"),
        os.path.join(hf_checkpoint_dir, "preprocessor_config.json")
    )

    # copy latest checkpoint txt
    latest_checkpoint = os.path.join(args.megatron_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest_checkpoint):
        shutil.copyfile(latest_checkpoint, os.path.join(hf_checkpoint_dir, "latest_checkpointed_iteration.txt"))
        print("copied latest_checkpointed_iteration.txt")

    with open(latest_checkpoint, 'r') as f:
        latest_iteration = f.read().strip()
    print(f"Converting model at iteration {latest_iteration}")

    # save metadata
    metadata = {
        "megatron_dir": args.megatron_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "tp": args.tp,
        "pp": args.pp,
        "latest_iteration": latest_iteration,
        "hf_checkpoint_dir": hf_checkpoint_dir
    }
    with open(os.path.join(hf_checkpoint_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

    loader = args.loader
    model_type = "LLaVA"
    saver = args.saver
    additional_args = "--use-dist-ckpt"
    tensor_model_parallel_size = args.tp
    pipeline_model_parallel_size = args.pp

    # Set environment variable
    os.environ["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    env_vars = {
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }
    env_args = ""
    for key, value in env_vars.items():
        env_args += f" --env {key}={value}"

    # Command to execute
    command = ["singularity", "exec", "--nv"] + env_args.split() + [
        args.sif_path,
        "python", "tools/checkpoint/convert.py",
        "--model-type", model_type,
        "--loader", loader,
        "--saver", saver,
        "--save-dir", hf_checkpoint_dir,
        "--load-dir", args.megatron_dir,
        "--megatron-path", "../Megatron-LM",
        "--tensor-model-parallel-size", str(tensor_model_parallel_size),
        "--pipeline-model-parallel-size", str(pipeline_model_parallel_size),
        "--hf-tokenizer-path", args.tokenizer_dir,
        additional_args
    ]

    # Execute the command
    subprocess.run(command)
    print("saved converted model to", hf_checkpoint_dir)
