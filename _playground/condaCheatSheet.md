# Conda and CUDA Cheatsheet

## Introduction
This guide provides essential commands and tips for managing Conda environments and setting up CUDA for GPU-accelerated machine learning workflows. It is designed to help data scientists, machine learning practitioners, and developers efficiently manage Conda environments and integrate CUDA for optimal performance.

## Table of Contents
- [Conda and CUDA Cheatsheet](#conda-and-cuda-cheatsheet)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installing Packages](#installing-packages)
  - [Creating Conda Environments](#creating-conda-environments)
    - [Basic Environment Creation](#basic-environment-creation)
    - [Example: Environment with Python 3.8](#example-environment-with-python-38)
  - [Using a Conda Configuration File](#using-a-conda-configuration-file)
    - [Create Environment from YAML](#create-environment-from-yaml)
    - [Export Environment to YAML](#export-environment-to-yaml)
  - [Setting up CUDA for TensorFlow](#setting-up-cuda-for-tensorflow)
    - [Install CUDA Toolkit](#install-cuda-toolkit)
    - [Verify CUDA Installation](#verify-cuda-installation)
  - [Checking TensorFlow Versions and GPUs](#checking-tensorflow-versions-and-gpus)
    - [Check TensorFlow Version](#check-tensorflow-version)
    - [Verify GPU Detection](#verify-gpu-detection)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues and Fixes](#common-issues-and-fixes)
  - [Helpful Commands](#helpful-commands)
    - [Delete a Conda Environment](#delete-a-conda-environment)
    - [Register Conda Environment for Jupyter](#register-conda-environment-for-jupyter)
    - [Update Conda Environment](#update-conda-environment)
  - [References](#references)

## Installing Packages
To install packages in your new Conda environment, use the following commands:

```bash
# Basic packages
conda install pip boto3 pandas ipykernel
conda install -c conda-forge opencv
conda install -c conda-forge python-dotenv
conda install matplotlib pyyaml

# CPU-only TensorFlow version
conda install -c conda-forge tensorflow

# GPU TensorFlow version
conda install -c conda-forge tensorflow-gpu

# Additional packages
conda install -c conda-forge scikit-learn
conda install -c conda-forge keras-tuner
```

## Creating Conda Environments
### Basic Environment Creation

```bash
# Create a new Conda environment with a specific Python version
conda create -n my_env python=3.9

# Activate the environment
conda activate my_env
```

### Example: Environment with Python 3.8
```bash
# Create a new environment with Python 3.8
conda create -n tf_env python=3.8

# Activate the environment
conda activate tf_env

# Install required packages
conda install tensorflow keras-tuner pip boto3 pandas ipykernel matplotlib pyyaml python-dotenv numpy
pip install datetime
pip install git+https://github.com/yaledhlab/vggface.git
```

## Using a Conda Configuration File
You can define a YAML configuration file to create a Conda environment. Here is an example:

**environment.yml**:
```yaml
name: my_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - tensorflow
  - keras-tuner
  - pip
  - boto3
  - pandas
  - ipykernel
  - matplotlib
  - pyyaml
  - python-dotenv
  - numpy
  - pip:
    - datetime
```

### Create Environment from YAML
```bash
conda env create -f environment.yml
```

### Export Environment to YAML
```bash
conda env export > environment.yml
```

## Setting up CUDA for TensorFlow
Ensure compatibility between TensorFlow, CUDA, and cuDNN versions. Use the table below as a reference:

| TensorFlow Version | CUDA Version | cuDNN Version |
|--------------------|--------------|---------------|
| 2.9               | 11.2         | 8.1           |
| 2.10              | 11.2         | 8.1           |

To determine the compatibility between TensorFlow versions and specific CUDA and cuDNN versions, you can refer to the following resources:
	1.	TensorFlow’s Official Installation Guide:
	•	The [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip) provides detailed instructions and [compatibility information](https://www.tensorflow.org/install/source#tested_build_configurations) for various TensorFlow versions, including the required CUDA and cuDNN versions. ￼
	2.	NVIDIA cuDNN Support Matrix:
	•	NVIDIA’s [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html) outlines the compatibility between different cuDNN versions and CUDA Toolkit versions. ￼

By consulting these resources, you can identify the appropriate CUDA and cuDNN versions compatible with your desired TensorFlow version, ensuring a smooth setup for your machine learning environment.

Check the CUDA version:

```zsh
cat /usr/local/cuda/version.txt
```

and cuDNN version:

```zsh
grep CUDNN_MAJOR -A 2 /usr/local/cuda/include/cudnn.h
```

### Install CUDA Toolkit
```bash
conda install -c conda-forge cudatoolkit=11.2
```

### Verify CUDA Installation
```bash
nvcc --version
```

## Checking TensorFlow Versions and GPUs
### Check TensorFlow Version
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Verify GPU Detection
```bash
# Check if TensorFlow can detect GPUs
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check GPU details
nvidia-smi
```

## Troubleshooting
### Common Issues and Fixes
- **GPU Not Detected**:
  - Ensure the correct version of CUDA Toolkit is installed.
  - Verify `nvidia-smi` outputs GPU details.

- **CUDA/TensorFlow Mismatch**:
  - Use the compatibility table to match versions.

- **Conflicts Between Conda and Pip**:
  - Prefer installing most packages via Conda. Use pip only when necessary and after installing Conda packages.

## Helpful Commands
### Delete a Conda Environment
```bash
conda env remove --name my_env
```

### Register Conda Environment for Jupyter
```bash
# Parse active environment name
export CURRENT_ENV_NAME=$(conda info | grep "active environment" | cut -d : -f 2 | tr -d ' ')

# Register the environment as a Jupyter kernel
python3 -m ipykernel install --user --name $CURRENT_ENV_NAME --display-name "user-env:($CURRENT_ENV_NAME)"
```

### Update Conda Environment
```bash
conda env update --file environment.yml --prune
```

## References
- [Conda Documentation](https://docs.conda.io/)
- [CUDA Toolkit Installation Guide](https://developer.nvidia.com/cuda-toolkit)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [NVIDIA System Management Interface (nvidia-smi)](https://developer.nvidia.com/nvidia-system-management-interface)

---

This cheatsheet is a living document. Feel free to update it as your workflows evolve!

