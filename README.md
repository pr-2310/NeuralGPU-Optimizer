# Project Training and Profiling Guide

This README outlines the procedure for training neural network models on NVIDIA GPUs, capturing performance metrics through profiling, and visualizing this data using roofline models.

## Getting Started

### Accessing NYU HPC and Dataset Preparation

1. **Open Command Prompt**: Start by opening your command prompt or terminal on your local machine.
2. **SSH into NYU HPC Gateway**:
ssh <NetID>@gw.hpc.nyu.edu
3. **Connect to Greene Cluster**:
ssh <NetID>@greene.hpc.nyu.edu
4. **Access ImageNet Dataset**: Obtain access to the ImageNet dataset on HPC and create a manageable subset for your experiments.
5. **Mount Subset on Burst**: Ensure the created subset is mounted on Burst for efficient access during training.

### Environment Setup

1. **SSH into Burst**:
ssh burst

2. **Prepare the Environment**: Load necessary modules for GPU access and singularity containers as per HPC documentation.

### Repository Cloning and Profiling Preparation

1. **Clone PyTorch Examples**:
git clone https://github.com/pytorch/examples.git
2. **Navigate to ImageNet Directory**:
cd examples/imagenet
3. **Modify for Profiling**: Integrate profiling commands into the training script as needed for detailed analysis.

### Profiling Execution

1. **Run Profiling Command**: Example for ResNet18 on V100 GPU:
ncu --profile-from-start off --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum --csv --page raw --log-file resnet18-v100.csv --target-processes all python main.py --arch resnet18 --epochs 1 --batch-size 4 --dummy --gpu 0
Adjust commands for other models (e.g., AlexNet) and GPUs (e.g., A100) as needed.

### Retrieving CSV File

1. **Download CSV File**: Use `scp` to transfer the CSV file from HPC to your local system:
scp <NetID>@greene.hpc.nyu.edu:/path/to/resnet18-v100.csv /local/path

### Data Analysis with Roofline Modeling

1. **Upload CSV to Google Colab**: Transfer the CSV files to Google Colab for analysis.
2. **Plot Roofline Model**: Utilize existing or create new Colab notebooks to visualize the performance data using roofline models.

This process facilitates a detailed comparison of neural network model performances across different hardware, enabling the identification of optimization opportunities.
