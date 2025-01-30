# Model training overview

![img.png](assets/img.png)

# Setup

## Build Singularity (Apptainer) Image
We use the Singularity environment to manage the environment for this project.

### How to build the Singularity image
- build singularity image from the recipe file (this will take about 10~20 minutes)
- this image is based on the docker image `nvcr.io/nvidia/pytorch:24.05-py3`
  - details about the base docker image: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.html
```shell
$ singularity build --nv --fakeroot \
  --disable-cache <path-to-singularity-image.sif> \
  environments/recipe_latest.def
```

# Prepare Dataset
Dataset should be prepared in the `megatron-energon` format.
Please refer to the megatron-energon's official documentation [here](https://nvidia.github.io/Megatron-Energon/index.html) for more details.

`megatron-energon` is a dataset format based on `webdataset`.
The dataset is stored in tar files, where each tar file contains multiple samples.

Each tar file must contain three files per sample:
- `[idx].jpg`: image file
- `[idx].txt`: answer text
- `[idx].json`: question text; e.g., `{"question": "What is this?"}`

tar files should be saved like below
```
<save-dir>/
└── energon/
    ├── 000000.tar
    ├── 000001.tar
    ├── ...
    └── 000999.tar
```

After creating tar files, you need to run `energon prepare` command to format the dataset in energon format
```shell
$ cd <save-dir>/energon
$ energon prepare ./

Found 5 tar files in total. The first and last ones are:
    - cor/dataset_part_1.tar
    - cor/dataset_part_5.tar
    If you want to exclude some of them, cancel with ctrl+c and specify an exclude filter in the command line.
    Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
    Indexing shards  [####################################]  5/5
    Sample 0, keys:
    - jpg
    - txt
    - json
Json content of sample 0 of cor/dataset_part_1.tar:
{
   "question": "Consider the image carefu...",
   "id": 0,
   "dataset_name": "cococaption",
   "original_id": 2,
   "original_image_id": 481464
}
Sample 1, keys:
 - jpg
   - txt
   - json
Json content of sample 1 of cor/dataset_part_1.tar:
{
    "question": "Provide a structured brea...",
    "id": 1,
    "dataset_name": "cococaption",
    "original_id": 4,
    "original_image_id": 514299
}
Found the following part types in the dataset: txt, json, jpg
Do you want to create a dataset.yaml interactively? [Y/n]: Y
The following dataset classes are available:
   0. CaptioningWebdataset
   1. CrudeWebdataset
   2. ImageClassificationWebdataset
   3. ImageWebdataset
   4. InterleavedWebdataset
   5. MultiChoiceVQAWebdataset
   6. OCRWebdataset
   7. SimilarityInterleavedWebdataset
   8. TextWebdataset
   9. VQAOCRWebdataset
   10. VQAWebdataset
   11. VidQAWebdataset
   Please enter a number to choose a class: 10
   The dataset you selected uses the following sample type:

@dataclass
class VQASample(Sample):
    """Sample type for visual question answering."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The context/question for the image
    context: str

    #: The possible answers. Not set for testing.
    answers: Optional[List[str]] = None
    #: The weights of the possible answers. Optionally available.
    answer_weights: Optional[torch.Tensor] = None

Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y

For each field, please specify the corresponding name in the WebDataset.
Available types in WebDataset: txt, json, jpg
Leave empty for skipping optional field
You may also access json fields e.g. by setting the field to: json[field][field]
You may also specify alternative fields e.g. by setting to: jpg,png
Please enter the field_map for VQAWebdataset:
Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
Please enter a webdataset field name for 'context' (<class 'str'>): json[question]
Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): txt
Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):
Done
``` 
  
This will create `.nv-meta` directory containing the metadata files for the dataset.


# Build Llava Model in Huggingface format
First, you need to build the model in Huggingface format. This can be done using the following script:
```shell
$ singularity exec <singularity-image.sif> \
  python scripts/convert/build_llava_jp.py \
    --save-dir <directory-to-save-model> \
    --text-model <huggingface model tag> [--vocab-size-padding]
```

# Convert Huggingface Model to Megatron Model
In order to train the model using Megatron, you need to convert the Huggingface model to the Megatron format.
This can be done using the following script:
```shell
$ python scripts/convert/convert_hf_to_megatron.py \
  --tp <tp num> \
  --pp <pp num> \
  --hf-dir <directory-to-load-model> \
  --megatron-dir <directory-to-save-model> \
  --sif-path <singularity.sif>
```


# Training

## Run training script
- Example
```shell
$ python scripts/train/run.py -g 8 \
  --save \
  --name <run_name> \
  --megatron-load-dir <megatron_moddel_dir> \
  --tokenizer-load-dir <hf_model_dir> \
  --save-dir <save_dir> \
  --sif-path <singularity.sif> \
  --data-path <prepared-dataset-path> \
  --tp 2 \
  --pp 1
```

### GPU-related arguments

- `--gpus, -g`: Number of GPUs to use. Default is -1, which typically means use all available GPUs.
- `--gpu-ids`: Comma-separated list of GPU IDs to use. An empty string means there's no specific restriction on GPU selection. In that case, the script will use GPUS from 0 to the number of GPUs specified by --gpus.


### Logging arguments

- `--save`: If enabled, saves the model to the specified directory. Note: optimizer states are not saved by default.
- `--name`: The name of the experiment, used for organizing saved models or logs.
- `--wandb`: Enables logging with Weights & Biases.
- `--log-interval`: Sets the interval (in terms of training iterations) at which logs are generated. default is 1.


### Save/Load arguments

- `--megatron-load-dir`: Specifies the directory to load the Megatron model checkpoint from.
- `--tokenizer-load-dir`: Specifies the directory to load the tokenizer model from.
- `--save-dir`: Directory where models, tokenizer, and optimizer states are saved.
- `--sif-path`: Singularity.sif
- `--save-optim`: Enables saving the state of the optimizer, useful for resuming training exactly where it left off.
- `--load-optim`: Loads the state of the optimizer if resuming training from a previous state.
- `--init`: If true, initializes the model from scratch rather than loading from a checkpoint.
- `--save-interval`: Sets the interval (in terms of training iterations) at which the model state is saved.
- `--skip-train-iteration-range`: Allows skipping specific ranges of training iterations, specified as one or more space-separated ranges. e.g., 100-110 300-400


### Model parallel arguments

- `--tp`: Number of tensor model parallelism segments. Default is 1.
- `--pp`: Number of pipeline model parallelism stages. Default is 1.
- `--micro-batch-size`: The size of each micro batch of data.
- `--global-batch-size`: The total size of the batch distributed across all parallel workers.


### Distributed settings

- `--num-workers`: Number of worker processes per node.
- `--node`: Total number of nodes in the distributed setup.
- `--rank, -r`: Rank of the current node within the distributed setup.
- `--master-addr, -m`: IP address of the master node to which all worker nodes connect.
- `--master-port, -p`: Port on the master node used for communication.
- `--backend`: Backend used for distributed training, typically 'nccl'. Other options include 'gloo' and 'mpi'.
- `--interface`: Network interface used for communication, such as 'eth0', 'ibp75s0', etc.
- `--rdzv-port`: Port used for rendezvous in distributed setups. If you want to run multiple jobs on the same node, you need to specify different ports.
- `--dist-optim`: Enables a distributed optimizer for training across multiple nodes.


### Tokenizer-related arguments

- `--image-token-id (optional)`: Token ID used to represent images in the tokenizer.
- `--pad-token-id (optional)`: Token ID used to represent padding in the tokenizer.


### Training settings

- `--freeze-lm`: Freezes the weights of the language model during training. For stage 1 training, this is typically set to True.
- `--stage`: Specifies the training stage, useful for multi-stage training setups.
- `--lang`: Main language of the model; typically 'jp' (Japanese) or 'en' (English).


### SLURM arguments
Note: These arguments are automatically set when running under SLURM.

- `--slurm-job-id`: Job ID assigned by SLURM, automatically set when running under SLURM.
- `--slurm-output-name`: The name used for output files generated by the SLURM job.


### Other arguments

- `--fp8`: Enables the use of FP8 precision.
- `--no-add-class-token`: Disables the addition of a class token to the outputs of the vision encoder.


### Debug arguments

- `--debug`: Enables debug mode, providing more verbose output for troubleshooting.
- `--deterministic`: Forces operations to be deterministic, which can aid in debugging but may reduce performance.

## Example command
- stage 1 (tp1, pp1)
```shell
$ python scripts/train/run.py \
  -g 1 --tp 1 --pp 1 --node 1 \
  --name llavajp1b-tp1-pp1-bf16-stage1 \
  --megatron-load-dir <your-megatron-model-path> \
  --save-dir <your-model-save-path> \
  --tokenizer-load-dir <hf-tokenizer-path> \
  --sif-path ./demo.sif \
  --stage 1 --freeze-lm \
  --save-interval 2000 --wandb
```

- stage 2 (tp4, pp1)
```shell
$ python scripts/train/run.py -g 8 --tp 4 --pp 1 --node 1 \
  --name llavajp13b-tp4-pp1-dp2-stage2 \
  --megatron-load-dir <your-megatron-model-path> \
  --save-dir <your-model-save-path> \
  --tokenizer-load-dir <hf-tokenizer-path> \
  --sif-path ./demo.sif \
  --stage 2 \
  --save-interval 2000 --wandb
```


# Convert Megatron Model to Huggingface Model
For evaluation, TP/PP conversion, etc., you may want to convert the Megatron model back to the Huggingface format.
This can be done using the following script:

```shell
$ python scripts/convert/convert_megatron_to_hf.py \
  --tp <tp num> \
  --pp <pp num> \
  --megatron-dir <directory-to-load-model> \
  --hf-dir <directory-to-save-model> \
  --tokenizer-dir <directory-to-load-tokenizer-model> \
```