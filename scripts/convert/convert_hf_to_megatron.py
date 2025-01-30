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
    parser.add_argument('--tp', type=int, default=1, help='Number of tensor model parallel')
    parser.add_argument('--pp', type=int, default=1, help='Number of pipeline model parallel')
    parser.add_argument(
        '--hf-dir', type=str,
        default="",
        help='Huggingface checkpoint directory'
    )
    parser.add_argument(
        '--megatron-dir', type=str,
        default="",
        help='Megatron checkpoint directory'
    )
    parser.add_argument(
        '--sif-path', type=str,
        help='Singularity file path'
    )
    parser.add_argument('--bind', type=str, help='Bind path; e.g., /data:/data')
    args = parser.parse_args()

    # Configuration variables
    hf_checkpoint_dir = args.hf_dir
    megatron_dir = args.megatron_dir
    os.makedirs(megatron_dir, exist_ok=True)
    print(f"MEGATRON_CHECKPOINT_DIR: {megatron_dir}")
    print(f"HF_CHECKPOINT_DIR: {hf_checkpoint_dir}")

    loader = "llava_hf"
    model_type = "LLaVA"
    saver = "megatron_llava"
    additional_args = "--use-dist-ckpt"
    tensor_model_parallel_size = args.tp
    pipeline_model_parallel_size = args.pp

    # Bind paths
    bind_mapping = [args.bind]

    bind_args = ""
    for bind in bind_mapping:
        bind_args += f" -B {bind}"

    # Replace the original path with the new bound path
    bind_megatron_checkpoint_dir = replace_path(megatron_dir, bind_mapping)
    bind_hf_checkpoint_dir = replace_path(hf_checkpoint_dir, bind_mapping)

    # Command to execute
    command = ["singularity", "exec", "--nv"] + bind_args.split() + [
        args.sif_path,
        "python", "tools/checkpoint/convert.py",
        "--model-type", model_type,
        "--loader", loader,
        "--saver", saver,
        "--save-dir", bind_megatron_checkpoint_dir,
        "--load-dir", bind_hf_checkpoint_dir,
        "--target-tensor-parallel-size", str(tensor_model_parallel_size),
        "--target-pipeline-parallel-size", str(pipeline_model_parallel_size),
        "--tokenizer-model", bind_hf_checkpoint_dir,
        additional_args
    ]

    # Execute the command
    print(command)
    result = subprocess.run(
        command,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "0",
            "APPTAINERENV_CUDA_VISIBLE_DEVICES": "0",
        },
    )

    if result.returncode != 0:
        print("Failed to convert the model.")
    else:
        print("saved converted model to", megatron_dir)
        # copy config file from hf to megatron
        shutil.copyfile(
            os.path.join(hf_checkpoint_dir, "config.json"),
            os.path.join(megatron_dir, "config.json")
        )
        # copy text_config.json from hf to megatron
        shutil.copyfile(
            os.path.join(hf_checkpoint_dir, "text_config.json"),
            os.path.join(megatron_dir, "text_config.json")
        )
        # copy preprocessor_config.json from hf to megatron
        shutil.copyfile(
            os.path.join(hf_checkpoint_dir, "preprocessor_config.json"),
            os.path.join(megatron_dir, "preprocessor_config.json")
        )
        # save metadata
        metadata = {
            "megatron_dir": megatron_dir,
            "hf_dir": hf_checkpoint_dir,
            "tp": args.tp,
            "pp": args.pp
        }
        with open(os.path.join(megatron_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
