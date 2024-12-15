# Conda and CUDA Cheatsheet

## Introduction
This guide provides essential commands and tips for managing Conda environments and setting up CUDA for GPU-accelerated machine learning workflows. It is designed to help data scientists, machine learning practitioners, and developers efficiently manage Conda environments and integrate CUDA for optimal performance.

## Table of Contents
- [Conda and CUDA Cheatsheet](#conda-and-cuda-cheatsheet)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Conda Cheatsheet](#conda-cheatsheet)
    - [__Installing Packages__](#installing-packages)
    - [__Creating Conda Environments__](#creating-conda-environments)
      - [__Basic Environment Creation__](#basic-environment-creation)
      - [__Example: Environment with Python 3.8__](#example-environment-with-python-38)
      - [__Using a Conda Configuration File__](#using-a-conda-configuration-file)
      - [__Create Environment from YAML__](#create-environment-from-yaml)
      - [__Export Environment to YAML__](#export-environment-to-yaml)
      - [__Delete a Conda Environment__](#delete-a-conda-environment)
      - [__Register Conda Environment for Jupyter__](#register-conda-environment-for-jupyter)
      - [__Update Conda Environment__](#update-conda-environment)
  - [CUDA Cheatsheet](#cuda-cheatsheet)
    - [__Setting up CUDA for TensorFlow__](#setting-up-cuda-for-tensorflow)
    - [__Check the CUDA and cuDNN version__](#check-the-cuda-and-cudnn-version)
    - [__Install CUDA Toolkit__](#install-cuda-toolkit)
    - [__Verify CUDA Installation__](#verify-cuda-installation)
    - [__Checking TensorFlow Versions and GPUs__](#checking-tensorflow-versions-and-gpus)
      - [__Check TensorFlow and Keras Versions__](#check-tensorflow-and-keras-versions)
      - [__Verify GPU Detection__](#verify-gpu-detection)
    - [__Troubleshooting__](#troubleshooting)
      - [__Common Issues and Fixes__](#common-issues-and-fixes)
    - [__References__](#references)

## Conda Cheatsheet

### __Installing Packages__

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

### __Creating Conda Environments__

#### __Basic Environment Creation__

```bash
# Create a new Conda environment with a specific Python version
conda create -n my_env python=3.9

# Activate the environment
conda activate my_env
```

#### __Example: Environment with Python 3.8__

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

#### __Using a Conda Configuration File__

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

#### __Create Environment from YAML__

```bash
conda env create -f environment.yml
```

#### __Export Environment to YAML__

```bash
conda env export > environment.yml
```

#### __Delete a Conda Environment__

```bash
conda env remove --name my_env
```

#### __Register Conda Environment for Jupyter__

```bash
# Parse active environment name
export CURRENT_ENV_NAME=$(conda info | grep "active environment" | cut -d : -f 2 | tr -d ' ')

# Register the environment as a Jupyter kernel
python3 -m ipykernel install --user --name $CURRENT_ENV_NAME --display-name "user-env:($CURRENT_ENV_NAME)"
```

#### __Update Conda Environment__

```bash
conda env update --file environment.yml --prune
```

## CUDA Cheatsheet

### __Setting up CUDA for TensorFlow__

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

### __Check the CUDA and cuDNN version__

```zsh
cat /usr/local/cuda/version.txt
```

and cuDNN version:

```zsh
grep CUDNN_MAJOR -A 2 /usr/local/cuda/include/cudnn.h
```

### __Install CUDA Toolkit__

```bash
conda install -c conda-forge cudatoolkit=11.2
```

### __Verify CUDA Installation__

```bash
nvcc --version
```

### __Checking TensorFlow Versions and GPUs__

#### __Check TensorFlow and Keras Versions__

```bash
python -c "import tensorflow as tf; print('Tensorflow Version: ', tf.__version__)"
python -c "import keras; print('Keras Version: ', keras.__version__)"
python -c "from tensorflow.keras.models import Sequential, Model"
```

#### __Verify GPU Detection__

```bash
# Check if TensorFlow can detect GPUs
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check GPU details
nvidia-smi
```

### __Troubleshooting__

#### __Common Issues and Fixes__

- **GPU Not Detected**:
  - Ensure the correct version of CUDA Toolkit is installed.
  - Verify `nvidia-smi` outputs GPU details.

- **CUDA/TensorFlow Mismatch**:
  - Use the compatibility table to match versions.

- **Conflicts Between Conda and Pip**:
  - Prefer installing most packages via Conda. Use pip only when necessary and after installing Conda packages.

### __References__

- [Conda Documentation](https://docs.conda.io/)
- [CUDA Toolkit Installation Guide](https://developer.nvidia.com/cuda-toolkit)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [NVIDIA System Management Interface (nvidia-smi)](https://developer.nvidia.com/nvidia-system-management-interface)

---

This cheatsheet is a living document. Feel free to update it as your workflows evolve!

