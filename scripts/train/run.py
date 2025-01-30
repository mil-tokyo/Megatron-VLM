import argparse
import json
import os
import shutil
import socket
import subprocess
import sys

import psutil
import yaml


def is_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False


def find_free_port_within_range(start_port, end_port):
    for port in range(start_port, end_port + 1):
        if is_port_free(port):
            return port
    raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")


def find_network_interface() -> str:
    # Find the network interface; e.g., ib**** or en****
    addrs = psutil.net_if_addrs()
    for interface in addrs:
        if interface.startswith('ib'):
            return interface
        elif interface.startswith('en'):
            return interface
    else:
        return ""


def replace_path(original_path: str, bind_mapping: list):
    for bind in bind_mapping:
        original_path = original_path.replace(bind.split(":")[0], bind.split(":")[1])
    return original_path


def print_with_bar(message=""):
    max_len = 30
    min_bar_len = 3
    message_len = len(message)

    if message == "":
        print("-" * max_len)
    else:
        bar_len = (max_len - message_len) // 2 - 2
        bar = "-" * max(min_bar_len, bar_len)
        print(f"{bar} {message} {bar}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--is-on-aws', action='store_true', help='If this script is running on AWS, set this flag')

    # gpu-related arguments
    parser.add_argument('--gpus', '-g', default=-1, type=int, help='Number of GPUs to use')
    parser.add_argument('--gpu-ids', type=str, default="", help='GPU IDs to use')

    # logging arguments
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--name', type=str, default='', help='Experiment name')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--log-interval', type=int, default=1, help='Log interval')

    # alert spike arguments
    parser.add_argument("--alert-spike", action="store_true", help="Alert loss spike")
    parser.add_argument("--spike-thr", type=float, default=2, help="Loss spike threshold")
    parser.add_argument("--slack-uid", type=str, default="", help="Slack user ID")

    # save/load arguments
    parser.add_argument('--megatron-load-dir', type=str, help='Megatron checkpoint directory')
    parser.add_argument('--tokenizer-load-dir', type=str, help='Tokenizer model directory')
    parser.add_argument('--save-dir', type=str, help='Save directory')
    parser.add_argument(
        '--sif-path', type=str,
        help='Singularity file path'
    )
    parser.add_argument(
        '--save-optim', action='store_true',
        help='Save optimizer; set to True if you want to resume training from the last checkpoint'
    )
    parser.add_argument(
        '--load-optim', action='store_true',
        help='Load optimizer; set to True if you want to resume training'
    )
    parser.add_argument('--init', action='store_true', help='Initialize the model from scratch')
    parser.add_argument('--save-interval', type=int, default=2000, help='Save interval')
    parser.add_argument('--non-persistent-save-interval', type=int, default=-1, help='Non-persistent save interval')
    parser.add_argument(
        '--skip-train-iteration-range', type=str, nargs='+', default=None,
        help='Iteration ranges to skip. The values are one or more dash-separated ranges. e.g., 101-200 251-300'
    )

    # model parallel arguments
    parser.add_argument('--tp', type=int, default=1, help='Number of tensor model parallel')
    parser.add_argument('--pp', type=int, default=1, help='Number of pipeline model parallel')
    parser.add_argument('--micro-batch-size', type=int, help='Micro batch size')
    parser.add_argument('--global-batch-size', type=int, help='Global batch size')

    # distributed settings
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--node', type=int, default=1, help='Number of nodes')
    parser.add_argument('--rank', '-r', type=int, default=-1, help='Node rank')
    parser.add_argument('--master-addr', '-m', type=str, default='192.168.174.130', help='Master address')
    parser.add_argument('--master-port', '-p', type=int, default=9902, help='Master port')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--interface', type=str, default='', help='Network interface; e.g., eth0, ibp75s0')
    parser.add_argument('--rdzv-port', type=int, default=29500, help='Rendezvous port')
    parser.add_argument('--dist-optim', action='store_true', help='Distributed optimizer')
    parser.add_argument('--dist-ckpt', action='store_true', help='Distributed checkpoint')

    # tokenizer-related arguments
    parser.add_argument('--image-token-id', type=int, default=-1, help='Image token ID')
    parser.add_argument('--pad-token-id', type=int, default=-1, help='Pad token ID')

    # train settings
    parser.add_argument('--freeze-lm', action='store_true', help='Freeze the language model')
    parser.add_argument('--stage', type=int, help='Stage number; 1 or 2')
    parser.add_argument('--lang', type=str, default='jp', help="llava main language; jp or en")
    parser.add_argument('--data-path', type=str, default=None, help="dataset")
    parser.add_argument('--train-iters', type=int, default=None, help="Number of training iterations")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--loss-scale', type=int, help="Loss scale")
    parser.add_argument('--adam-beta2', type=float, default=0.999, help="Adam beta2")

    # slurm arguments (will automatically be set by the SLURM job)
    parser.add_argument('--slurm-job-id', type=int, default=-1, help='SLURM job ID')
    parser.add_argument('--slurm-output-name', type=str, default='', help='SLURM output name')

    # other arguments
    parser.add_argument('--fp8', action='store_true', help='Use FP8')
    parser.add_argument('--fused-attn', action='store_true', help='Use fused attention')
    parser.add_argument('--flash-attn', action='store_true', help='Use flash attention')
    parser.add_argument('--no-add-class-token', action='store_true', help='Do not add class token')
    parser.add_argument('--use-fixed-attn-mask', action='store_true', help='Use fixed attention mask; WIP')
    parser.add_argument('--seq-length', type=int, default=1024, help='Sequence length')

    # debug arguments
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic mode')

    args = parser.parse_args()

    is_aws = args.is_on_aws

    tp_num = args.tp if not args.debug else 1
    pp_num = args.pp if not args.debug else 1
    total_parallel = tp_num * pp_num

    if args.image_token_id == -1 or args.pad_token_id == -1:
        # load tokenizer
        tokenizer_config_path = os.path.join(args.tokenizer_load_dir, "tokenizer_config.json")
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        added_tokens = tokenizer_config["added_tokens_decoder"]
        # added_tokens: Dict[str, Any]
        for token_id, token_info in added_tokens.items():
            if token_info["content"] == "<image>" and args.image_token_id == -1:
                args.image_token_id = int(token_id)
            if token_info["content"] == "<pad>" and args.pad_token_id == -1:
                args.pad_token_id = int(token_id)
        print(f"image_token_id: {args.image_token_id}, pad_token_id: {args.pad_token_id}")

    # set up GPUs
    if args.gpu_ids:
        if args.gpus == -1:
            num_gpus = len(args.gpu_ids.split(","))
        else:
            assert len(args.gpu_ids.split(",")) == args.gpus, "Number of GPUs does not match"
            num_gpus = args.gpus
        visible_devices = args.gpu_ids
    else:
        num_gpus = total_parallel if args.gpus == -1 else args.gpus
        visible_devices = ",".join([str(i) for i in range(num_gpus)])
    print(f"tp_num: {tp_num}, pp_num: {pp_num}, total_parallel: {total_parallel}")
    print(f"num_gpus: {num_gpus}, visible_devices: {visible_devices}")

    if is_port_free(args.master_port):
        master_port = args.master_port
    else:
        if args.node > 1:
            free_port = find_free_port_within_range(9800, 9999)
            raise RuntimeError(f"Port {args.master_port} is already in use. "
                               f"Please specify a different port; e.g., --master-port {free_port}")
        else:
            print(f"Port {args.master_port} is already in use. Finding a free port...")
            master_port = find_free_port_within_range(9800, 9999)
    print(f"master_port: {master_port}")

    exp_name = f"{args.name}-" if args.name else ""
    exp_name += f"tp{tp_num}-pp{pp_num}-gpus{num_gpus}"
    if args.node > 1:
        exp_name += f"-nodes{args.node}"

    # Set up paths
    megatron_checkpoint_dir = args.megatron_load_dir
    megatron_save_dir = f"{args.save_dir}/{exp_name}"
    tokenizer_model = args.tokenizer_load_dir
    dataset_path = args.data_path

    bind_mapping = [
        f"{dataset_path}:/datasets",
    ]

    bind_args = ""
    for bind in bind_mapping:
        bind_args += f" -B {bind}"

    # replace original path with the new bound path
    print_with_bar("original paths")
    print(f"loading from {megatron_checkpoint_dir}")
    print(f"saving to {megatron_save_dir}")
    print(f"loading tokenizer from {tokenizer_model}")
    os.makedirs(megatron_save_dir, exist_ok=True)
    print_with_bar()

    # copy the config from tokenizer directory to megatron directory
    print()
    config_files = [xf for xf in os.listdir(f"{tokenizer_model}") if xf.endswith("config.json")]
    for config_file in config_files:
        if not os.path.exists(f"{megatron_checkpoint_dir}/{config_file}"):
            shutil.copyfile(f"{tokenizer_model}/{config_file}", f"{megatron_checkpoint_dir}/{config_file}")
        if not os.path.exists(f"{megatron_save_dir}/{config_file}"):
            shutil.copyfile(f"{tokenizer_model}/{config_file}", f"{megatron_save_dir}/{config_file}")
        print(f"copied {config_file} to {megatron_checkpoint_dir}, {megatron_save_dir}")

    bind_megatron_checkpoint_dir = replace_path(megatron_checkpoint_dir, bind_mapping)
    bind_megatron_save_dir = replace_path(megatron_save_dir, bind_mapping)
    bind_tokenizer_model = replace_path(tokenizer_model, bind_mapping)
    print_with_bar("after replacing paths")
    print(f"MEGATRON_CHECKPOINT_DIR: {bind_megatron_checkpoint_dir}")
    print(f"TOKENIZER_MODEL: {bind_tokenizer_model}")
    print_with_bar()

    # load dataset info
    with open(f"{dataset_path}/.nv-meta/.info.yaml", "r") as f:
        dataset_info = yaml.safe_load(f)
    with open(f"{dataset_path}/.nv-meta/split.yaml", "r") as f:
        split_info = yaml.safe_load(f)
    train_splits = split_info["split_parts"]["train"]
    valid_splits = split_info["split_parts"]["val"]

    shard_counts = dataset_info["shard_counts"]
    train_size = sum([shard_counts[s] for s in train_splits])
    valid_size = sum([shard_counts[s] for s in valid_splits])

    if args.stage == 1:
        global_batch_size = 256 if args.global_batch_size is None else args.global_batch_size
        micro_batch_size = 16 if args.micro_batch_size is None else args.micro_batch_size
        learning_rate = 1.0e-3 if args.lr is None else args.lr
    else:
        global_batch_size = 128 if args.global_batch_size is None else args.global_batch_size
        micro_batch_size = 1 if args.micro_batch_size is None else args.micro_batch_size
        learning_rate = 2.0e-5 if args.lr is None else args.lr
    if args.debug:
        global_batch_size = 1
        micro_batch_size = 1
        learning_rate = 1.0e-4

    if args.train_iters is None or args.train_iters < 0:
        train_iter = train_size // global_batch_size
    else:
        train_iter = args.train_iters

    os.environ["CHECKPOINT_PATH"] = "./output"

    # Model parallel arguments
    model_parallel_args = [
        "--tensor-model-parallel-size", str(tp_num),
        "--pipeline-model-parallel-size", str(pp_num)
    ]
    if args.dist_optim:
        model_parallel_args.append("--use-distributed-optimizer")
    if args.dist_ckpt:
        model_parallel_args.append("--use-dist-ckpt")

    # model load arguments
    model_load_args = [
        "--load", bind_megatron_checkpoint_dir,
    ] if not args.init else []
    model_load_args += [
        "--config-json-path", f"{bind_megatron_checkpoint_dir}/config.json",
    ]

    # GPT arguments
    with open(f"{megatron_checkpoint_dir}/text_config.json", "r") as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    num_query_groups = config.get("num_key_value_heads", 1)

    gpt_args = [
        "--num-layers", str(num_layers),
        "--hidden-size", str(hidden_size),
        "--num-attention-heads", str(num_attention_heads),
        "--seq-length", str(args.seq_length),  # 128 or 1024
        "--max-position-embeddings", str(args.seq_length),  # 128 or 1024
        "--micro-batch-size", str(micro_batch_size),
        "--global-batch-size", str(global_batch_size),
        "--lr", str(learning_rate),
        "--train-iters", str(train_iter),
        "--multimodal-train-samples", str(train_size),
        "--lr-decay-style", "cosine",
        "--min-lr", "1.0e-8",
        "--weight-decay", "1e-1",
        "--lr-warmup-fraction", ".03",
        "--clip-grad", "1.0",
        "--freeze-ViT",
        "--bf16",
        "--encoder-pipeline-model-parallel-size", "1"  # encoder pipeline size must be 1,
    ]
    if args.fp8:
        gpt_args += ["--fp8-format", "hybrid"]
    if args.freeze_lm or args.stage == 1:
        gpt_args += ["--freeze-LM"]
    if not args.no_add_class_token:
        gpt_args += ["--add-class-token"]
    if not args.deterministic:
        gpt_args += ["--use-flash-attn"]
    if args.use_fixed_attn_mask:
        gpt_args += ["--use-fixed-attn-mask"]
    if args.loss_scale is not None:
        gpt_args += ["--loss-scale", "16384"]

    # Image arguments
    with open(f"{megatron_checkpoint_dir}/config.json", "r") as f:
        config = json.load(f)
    image_size = config['vision_config']['image_size']
    patch_size = config['vision_config']['patch_size']
    img_args = [
        "--img-h", str(image_size),
        "--img-w", str(image_size),
        "--patch-dim", str(patch_size),
    ]

    # Data arguments
    data_args = [
        "--split", "100,0,0",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--data-path", "scripts/train/sft_dataset.yaml",
        "--valid-path", "scripts/train/sft_dataset.yaml",
        "--prompt-path", "scripts/train/manual_prompts.json",
        "--dataloader-type", "external",
        "--training-stage", str(args.stage),
    ]
    if args.skip_train_iteration_range is not None:
        data_args.extend([
            "--skip-train-iteration-range", *args.skip_train_iteration_range
        ])

    # Training arguments
    train_args = [
        "--no-check-for-nan-in-loss-and-grad",
        "--distributed-backend", args.backend,
        "--distributed-timeout-minutes", "120",
    ] if args.node > 1 else [
        "--distributed-backend", "nccl"
    ]
    if args.deterministic:
        train_args.extend([
            "--deterministic-mode",
        ])

    # Output arguments
    output_args = [
        "--log-interval", "2",
        "--save-interval", str(args.save_interval) if not args.debug else str(train_iter * 10),
        "--eval-interval", str(train_iter * 10),  # no evaluation
        "--eval-iters", "2",
    ]
    if args.non_persistent_save_interval > 0:
        output_args.extend([
            "--non-persistent-save-interval", str(args.non_persistent_save_interval),
            "--non-persistent-ckpt-type", "global"
        ])
    if not args.save_optim:
        output_args.append("--no-save-optim")
    if not args.load_optim:
        output_args.extend(["--no-load-optim", "--no-load-rng"])

    if args.save_optim:
        output_args.extend(["--dataloader-save", bind_megatron_save_dir])

    if args.wandb:
        tags = []
        if args.stage == 1:
            tags.append("stage1")
        elif args.stage == 2:
            tags.append("stage2")

        if args.lang == 'jp':
            tags.append("llavajp")
        elif args.lang == 'en':
            tags.append("llava")

        tags.append(f"nodes{args.node}")
        tags.append(f"tp{tp_num}")
        tags.append(f"pp{pp_num}")
        tags.append(f"gpus{num_gpus}")
        if args.init:
            tags.append("init")

        log_args = [
            "--wandb-project", "megatron-llava",
            "--wandb-exp-name", exp_name,
            "--tensorboard-log-interval", str(args.log_interval),
            "--wandb-tags", ",".join(tags),
        ]
    else:
        log_args = []

    # distributed arguments
    if args.node > 1:
        torchrun_args = [
            "--master-addr", args.master_addr,
            "--master-port", str(master_port),
            "--nnodes", str(args.node),
            "--nproc-per-node", str(num_gpus),
        ]
        if args.rank > -1:
            torchrun_args.extend(["--node-rank", str(args.rank)])
    else:
        torchrun_args = [
            "--master-port", str(master_port),
            "--nnodes", "1",
            "--master-addr", "localhost",
            "--nproc-per-node", str(num_gpus),
        ]
        if args.rdzv_port:
            torchrun_args.extend(["--rdzv-endpoint", f"localhost:{args.rdzv_port}"])

    if args.alert_spike:
        alert_args = [
            "--alert-spike",
            "--spike-thr",
            str(args.spike_thr)
        ]
        if args.slack_uid:
            alert_args.extend([
                "--slack-uid",
                args.slack_uid
            ])
    else:
        alert_args = []

    # Environment variables for Singularity
    env_vars = {
        "APPTAINERENV_CUDA_VISIBLE_DEVICES": visible_devices,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MASTER_ADDR": "localhost",
    }
    if args.fused_attn:
        env_vars["NVTE_FUSED_ATTN"] = "1"
        env_vars["NVTE_FLASH_ATTN"] = "0"
    if args.flash_attn:
        env_vars["NVTE_FLASH_ATTN"] = "1"
        env_vars["NVTE_FUSED_ATTN"] = "0"

    if args.deterministic:
        deterministic_env_vars = {
            "NCCL_ALGO": "Tree",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
        }
        env_vars.update(deterministic_env_vars)

    if args.interface == "" and args.node > 1:
        args.interface = find_network_interface()
        print(f"Network interface: {args.interface}")
    dist_env_vars = {
        "GLOO_SOCKET_IFNAME": args.interface,
        # "NCCL_DEBUG": "INFO",
        # "NCCL_DEBUG_SUBSYS": "ALL",
        "NCCL_SOCKET_IFNAME": args.interface,
        "MASTER_ADDR": args.master_addr,
        "MASTER_PORT": str(master_port),
    }
    if args.node <= 1:
        # Remove socket interface variables
        dist_env_vars.pop("GLOO_SOCKET_IFNAME")
        dist_env_vars.pop("NCCL_SOCKET_IFNAME")

    if args.node > 1:
        # combine env_vars and dist_env_vars
        env_vars.update(dist_env_vars)
        if is_aws:
            efa_vars = {
                "FI_PROVIDER": "efa",
                "FI_EFA_USE_DEVICE_RDMA": "1",
                "NCCL_ALGO": "Ring,Tree",
                "NCCL_PROTO": "LL128",
                "NCCL_IB_DISABLE": "1",

                "FI_EFA_MR_CACHE_ENABLE": "1",  # slightly slower
                "FI_EFA_SET_CUDA_SYNC_MEMOPS": "0",  # slightly slower
                "FI_OFI_RXR_RX_COPY_UNEXP": "1",
                "FI_OFI_RXR_RX_COPY_OOO": "1",
                "FI_OFI_RXR_INLINE_MR_ENABLE": "1",

                "NCCL_TREE_THRESHOLD": str(10 * 4294967296),
                "NCCL_LAUNCH_MODE": "PARALLEL",
                "NCCL_NET_SHARED_COMMS": "0",
                "NCCL_BUFFSIZE": "8388608",
                # "BUCKET_CAP_MB": "512",

                # "NCCL_CROSS_NIC": "0",
                "NCCL_NSOCKS_PERTHREAD": "8",
                "NCCL_SOCKET_NTHREADS": "1",
                # "NCCL_DYNAMIC_CHUNK_SIZE": "524288",
                "NCCL_P2P_NET_CHUNKSIZE": "524288",
                # "NCCL_P2P_PCI_CHUNKSIZE": "524288",
                # "NCCL_P2P_NVL_CHUNKSIZE": "524288",
                # "NCCL_NET_GDR_LEVEL": "PIX",
                # "NCCL_P2P_PXN_LEVEL": "0",

                "FI_EFA_FORK_SAFE": "1",
                # "NCCL_TUNER_PLUGIN": "/usr/local/lib/libnccl-ofi-tuner.so",
                "NCCL_TUNER_PLUGIN": "/opt/aws-ofi-nccl/lib/libnccl-ofi-tuner.so",

                "OMP_NUM_THREADS": "12",

                "LD_LIBRARY_PATH": f"/opt/amazon/efa/lib64:/opt/aws-ofi-nccl/lib:{os.environ['LD_LIBRARY_PATH']}",
            }
            env_vars.update(efa_vars)

    env_args = ""
    for key, value in env_vars.items():
        env_args += f" --env {key}={value}"

    # Construct the command
    script = "scripts/train/pretrain_vlm.py"
    if args.debug:
        python_commands = ["python", "-m", "pdb", script]
    else:
        python_commands = ["torchrun"] + torchrun_args + [script]

    # pass the command to run this script as --command argument to the script
    executed_command = ' '.join(sys.argv)

    if args.slurm_job_id != -1:
        log_args.extend([
            "--slurm-job-id", str(args.slurm_job_id),
            "--slurm-output-name", args.slurm_output_name,
        ])

    command = (
        ["singularity", "exec", "--cleanenv", "--nv"] +
        env_args.split() + bind_args.split() +
        [args.sif_path] +
        python_commands + gpt_args + img_args + data_args + output_args +
        ["--num-workers", str(args.num_workers),
         "--untie-embeddings-and-output-weights",
         "--recompute-activations",
         "--recompute-granularity", "selective",
         "--adam-beta2", str(args.adam_beta2),
         # "--sequence-parallel",
         "--save", bind_megatron_save_dir,
         "--tokenizer-model", bind_tokenizer_model,
         "--exit-on-missing-checkpoint",
         "--image-token-id", str(args.image_token_id),
         "--pad-token-id", str(args.pad_token_id),
         "--command", executed_command,
         "--no-initialization",
         "--group-query-attention",
         "--num-query-groups", str(num_query_groups),
         "--env-params", " ".join([f"{k}={v}" for k, v in env_vars.items()]),
         ] +
        model_load_args +
        model_parallel_args + log_args + train_args + alert_args
    )

    # Print and execute the command
    print(f"\nExecuting command:\n\n{' '.join(command).strip()}\n")

    local_envs = {
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "NVIDIA_VISIBLE_DEVICES": visible_devices,
        "APPTAINERENV_CUDA_VISIBLE_DEVICES": visible_devices,
    }
    if is_aws and args.node > 1:
        local_envs.update(efa_vars)

    subprocess.run(
        command,
        env=local_envs,
    )
