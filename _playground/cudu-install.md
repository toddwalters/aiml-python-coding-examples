# Windows WSL CUDA Install and Configuration

## Monitoring GPU Usage

```zsh
clear; gpustat -a -c -f --watch
```

OR

```zsh
nvitop
```

## Reference Links

- [Windows Driver Download](https://www.nvidia.com/en-us/drivers/results/)
- [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
  - [GeForce RTX 3060 Family](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/)
- [CUDA Download - WSL-Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
  - [NVIDIA CUDA Installation Guide for Linux - Pre-Installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
  - [NVIDIA CUDA Installation Guide for Linux - Ubuntu Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#prepare-ubuntu)
  - [NVIDIA CUDA Installation Guide for Linux - Post-Installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
  - [NVIDIA CUDA cuda-samples GitHub Repo](https://github.com/NVIDIA/cuda-samples/tree/master)
    - [NVIDIA CUDA cuda-samples deviceQuery Utility](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery)
- [cuDNN 9.6.0 Downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
  - [Support Matrix GPU, CUDA Toolkit, and CUDA Driver Requirements](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html#f3)
- [Tensor RT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
  - [NVIDIA TensorRT 10.x Download](https://developer.nvidia.com/tensorrt/download/10x)


## Pre-Install Steps

### Remove previous NVIDIA installation
```shell
sudo apt autoremove \*nvidia\* --purge # 
```

### Update & upgrade
```shell
sudo apt update && sudo apt upgrade
```

## OS Details

```zsh
cat /etc/os-release
───────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       │ File: /etc/os-release
───────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1   │ PRETTY_NAME="Ubuntu 22.04.5 LTS"
   2   │ NAME="Ubuntu"
   3   │ VERSION_ID="22.04"
   4   │ VERSION="22.04.5 LTS (Jammy Jellyfish)"
   5   │ VERSION_CODENAME=jammy
   6   │ ID=ubuntu
   7   │ ID_LIKE=debian
   8   │ HOME_URL="https://www.ubuntu.com/"
   9   │ SUPPORT_URL="https://help.ubuntu.com/"
  10   │ BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
  11   │ PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
  12   │ UBUNTU_CODENAME=jammy
```

## Packages to Install

### Install CUDA Toolkit

- [CUDA Download - WSL-Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
  - [NVIDIA CUDA Installation Guide for Linux - Pre-Installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
  - [NVIDIA CUDA Installation Guide for Linux - Ubuntu Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#prepare-ubuntu)
  - [NVIDIA CUDA Installation Guide for Linux - Post-Installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
  - [NVIDIA CUDA cuda-samples GitHub Repo](https://github.com/NVIDIA/cuda-samples/tree/master)
    - [NVIDIA CUDA cuda-samples deviceQuery Utility](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery)

Follow the instruction listed above on installation of the CUDA Development kit, but it should look something like this:

1. wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
2. sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
3. wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
4. sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
5. sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
6. sudo apt-get update
7. sudo apt-get -y install cuda-toolkit-12-6

**NOTE:**  **DO NOT SET `LD_LIBRARY_PATH` variable.**  The Post-Installation instructions may tell you to do this, but in the case of WSL based installs, I found that setting `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu` **breaks the ability to run nvidia-smi command as non-root user.**

Also ensure that the CUDA Toolkit `bin` directory is listed at the beginning of the `PATH` environment variable list to ensure those binaries are found first.

### Install NVIDIA drivers

I am not sure this is actually required.  I would assume if this being installed the version that is to be installed should match the version of the NVDIA driver installed in Windows.  Howerever, I was able to make things work when I installed nvidia-driver-525 even though Driver Version: 566.36 is installed on the Windows side.  This makes me think installing these might not be required, but leaving it in for now.

```zsh
sudo apt install nvidia-driver-525
```

## CUDA Toolkit Install Verification

### python

Assuming you have a `conda` environment configured that has at least `tensorflow` installed you should be able to check visibility of the GPU via the following `python` command

```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### nvcc

```zsh
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

### nvidia-smi

```zsh
nvidia-smi

Mon Dec  9 14:49:50 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 561.09         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        On  |   00000000:01:00.0  On |                  N/A |
|  0%   34C    P8             15W /  170W |    1716MiB /  12288MiB |      8%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A        31      G   /Xwayland                                   N/A      |
+-----------------------------------------------------------------------------------------+
```

### deviceQuery Output

```zsh
cuda-samples/Samples/1_Utilities/deviceQuery  λ

./deviceQuery

./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3060"
  CUDA Driver Version / Runtime Version          12.6 / 12.6
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 12288 MBytes (12884377600 bytes)
  (028) Multiprocessors, (128) CUDA Cores/MP:    3584 CUDA Cores
  GPU Max Clock rate:                            1837 MHz (1.84 GHz)
  Memory Clock rate:                             7501 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 2359296 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.6, CUDA Runtime Version = 12.6, NumDevs = 1
Result = PASS
```

### bandwidthTest Output

```zsh
cuda-samples/Samples/1_Utilities/deviceQuery λ

../bandwidthTest/bandwidthTest

[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: NVIDIA GeForce RTX 3060
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     23.3

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     25.7

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     301.0

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

## cuDNN 9.6.0 Download & Install

- [cuDNN 9.6.0 Downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
  - [Support Matrix GPU, CUDA Toolkit, and CUDA Driver Requirements](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html#f3)

1. wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
2. sudo dpkg -i cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
3. sudo cp /var/cudnn-local-repo-ubuntu2204-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
4. sudo apt-get update
5. sudo apt-get -y install cudnn
6. sudo apt-get install libcudnn8 libcudnn8-dev

### Verify cuDNN Install

#### __Check Library Linking__

```zsh
ldconfig -p | grep cudnn
        libcudnn_ops.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_ops.so.9
        libcudnn_ops.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_ops.so
        libcudnn_heuristic.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_heuristic.so.9
        libcudnn_heuristic.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_heuristic.so
        libcudnn_graph.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_graph.so.9
        libcudnn_graph.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_graph.so
        libcudnn_engines_runtime_compiled.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9
        libcudnn_engines_runtime_compiled.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so
        libcudnn_engines_precompiled.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9
        libcudnn_engines_precompiled.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so
        libcudnn_cnn.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_cnn.so.9
        libcudnn_cnn.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_cnn.so
        libcudnn_adv.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_adv.so.9
        libcudnn_adv.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn_adv.so
        libcudnn.so.9 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn.so.9
        libcudnn.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libcudnn.so
```

#### __Test with a Deep Learning Framework__

```zsh
import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

python -c "import torch; print('PyTorch CUDA available: ', torch.cuda.is_available()); print('cuDNN enabled: ', torch.backends.cudnn.enabled)"

> PyTorch CUDA available:  True
> cuDNN enabled:  True
```

#### __Run a Sample Code Using cuDNN APIs (Advanced)__

```zsh
sudo apt-get update
sudo apt-get install libfreeimage-dev

cd /usr/src/cudnn_samples_v8/mnistCUDNN
sudo make
./mnistCUDNN

----------------------

Executing: mnistCUDNN
cudnnGetVersion() : 90600 , CUDNN_VERSION from cudnn.h : 90600 (9.6.0)
Host compiler version : GCC 11.4.0

There are 1 CUDA capable devices on your machine :
device 0 : sms 28  Capabilities 8.6, SmClock 1837.0 Mhz, MemSize (Mb) 12287, MemClock 7501.0 Mhz, Ecc=0, boardGroupID=0
Using device 0

Testing single precision
Loading binary file data/conv1.bin
Loading binary file data/conv1.bias.bin
Loading binary file data/conv2.bin
Loading binary file data/conv2.bias.bin
Loading binary file data/ip1.bin
Loading binary file data/ip1.bias.bin
Loading binary file data/ip2.bin
Loading binary file data/ip2.bias.bin
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 184784 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 2057744 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.060288 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.060416 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.078848 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.132096 time requiring 184784 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.177088 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.311296 time requiring 178432 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 128848 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 128000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 2450080 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.024576 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.114720 time requiring 128000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.130048 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.139264 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.143072 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.164928 time requiring 128848 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000
Loading image data/three_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 184784 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 2057744 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.006144 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.007008 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.007168 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.094208 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.105376 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.217088 time requiring 184784 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 128848 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 128000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 2450080 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.024320 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.106496 time requiring 128000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.122816 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.131808 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.136192 time requiring 128848 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.337152 time requiring 4656640 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006

Result of classification: 1 3 5

Test passed!

Testing half precision (math in single precision)
Loading binary file data/conv1.bin
Loading binary file data/conv1.bias.bin
Loading binary file data/conv2.bin
Loading binary file data/conv2.bias.bin
Loading binary file data/ip1.bin
Loading binary file data/ip1.bias.bin
Loading binary file data/ip2.bin
Loading binary file data/ip2.bias.bin
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 184784 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 2057744 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.065536 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.068608 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.071680 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.162816 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.228224 time requiring 184784 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.268128 time requiring 178432 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 64000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 1433120 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.030496 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.038912 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.117760 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.128800 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.130880 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.187360 time requiring 64000 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Resulting weights from Softmax:
0.0000001 1.0000000 0.0000001 0.0000000 0.0000563 0.0000001 0.0000012 0.0000017 0.0000010 0.0000001
Loading image data/three_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 184784 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 2057744 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.006144 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.009120 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.015360 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.228352 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.375776 time requiring 178432 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.391168 time requiring 184784 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnGetConvolutionForwardAlgorithm_v7 ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: -1.000000 time requiring 64000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: -1.000000 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: -1.000000 time requiring 2450080 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: -1.000000 time requiring 1433120 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.030720 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.030720 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.071680 time requiring 1433120 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.083040 time requiring 64000 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.123904 time requiring 4656640 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.124768 time requiring 2450080 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 6: -1.000000 time requiring 0 memory
^^^^ CUDNN_STATUS_NOT_SUPPORTED for Algo 3: -1.000000 time requiring 0 memory
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000714 0.0000000 0.0000000 0.0000000 0.0000000
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 1.0000000 0.0000154 0.0000000 0.0000012 0.0000006

Result of classification: 1 3 5

Test passed!
```

## Install NVIDIA TensorRT

- [Tensor RT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
  - [NVIDIA TensorRT 10.x Download](https://developer.nvidia.com/tensorrt/download/10x)

### Install TensorRT Packages

**NOTE:**  These packages may already have been installed as part of the conda environment set-up

```zsh
pip install tensorrt tensorrt_lean tensorrt_dispatch

--------------------

Successfully built tensorrt tensorrt_cu12 tensorrt-lean tensorrt_lean_cu12 tensorrt-dispatch tensorrt_dispatch_cu12
Installing collected packages: tensorrt_lean_cu12, tensorrt_dispatch_cu12, tensorrt_cu12, tensorrt-lean, tensorrt-dispatch, tensorrt
Successfully installed tensorrt-10.7.0 tensorrt-dispatch-10.7.0 tensorrt-lean-10.7.0 tensorrt_cu12-10.7.0 tensorrt_dispatch_cu12-10.7.0 tensorrt_lean_cu12-10.7.0
```

### Install TensorRT System Libraries

**NOTE:**  These packages may already have been installed as part of the conda environment set-up
**NOTE:**  If you are installing via `pip` then you shouldn't need to perform the steps listed below.  Follwing the process outlined below is probably only necessary in the event that you are trying to ensure the installation of very specific versions of the TensorRT System Libraries.

```zsh
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
export os="ubuntu2204"
export tag="10.7.0-cuda-12.6"
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt

---------------------------------------------------------

Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following packages were automatically installed and are no longer required:
  nsight-compute-2024.3.1 nsight-systems-2024.4.2
Use 'sudo apt autoremove' to remove them.
The following additional packages will be installed:
  libnvinfer-bin libnvinfer-dev libnvinfer-dispatch-dev libnvinfer-dispatch10 libnvinfer-headers-dev libnvinfer-headers-plugin-dev libnvinfer-lean-dev libnvinfer-lean10 libnvinfer-plugin-dev
  libnvinfer-plugin10 libnvinfer-samples libnvinfer-vc-plugin-dev libnvinfer-vc-plugin10 libnvinfer10 libnvonnxparsers-dev libnvonnxparsers10 python3-libnvinfer python3-libnvinfer-dev
  python3-libnvinfer-dispatch python3-libnvinfer-lean zlib1g-dev
The following NEW packages will be installed:
  libnvinfer-bin libnvinfer-dev libnvinfer-dispatch-dev libnvinfer-dispatch10 libnvinfer-headers-dev libnvinfer-headers-plugin-dev libnvinfer-lean-dev libnvinfer-lean10 libnvinfer-plugin-dev
  libnvinfer-plugin10 libnvinfer-samples libnvinfer-vc-plugin-dev libnvinfer-vc-plugin10 libnvinfer10 libnvonnxparsers-dev libnvonnxparsers10 python3-libnvinfer python3-libnvinfer-dev
  python3-libnvinfer-dispatch python3-libnvinfer-lean tensorrt zlib1g-dev
0 upgraded, 22 newly installed, 0 to remove and 22 not upgraded.
Need to get 164 kB/2731 MB of archives.
After this operation, 7045 MB of additional disk space will be used.
Do you want to continue? [Y/n] Y
Get:1 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 zlib1g-dev amd64 1:1.2.11.dfsg-2ubuntu9.2 [164 kB]
Get:2 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer10 10.7.0.23-1+cuda12.6 [1240 MB]
Get:3 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-lean10 10.7.0.23-1+cuda12.6 [8232 kB]
Get:4 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-plugin10 10.7.0.23-1+cuda12.6 [9874 kB]
Get:5 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-vc-plugin10 10.7.0.23-1+cuda12.6 [223 kB]
Get:6 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-dispatch10 10.7.0.23-1+cuda12.6 [213 kB]
Get:7 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvonnxparsers10 10.7.0.23-1+cuda12.6 [1324 kB]
Get:8 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-bin 10.7.0.23-1+cuda12.6 [461 kB]
Get:9 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-headers-dev 10.7.0.23-1+cuda12.6 [106 kB]
Get:10 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-dev 10.7.0.23-1+cuda12.6 [1246 MB]
Get:11 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-dispatch-dev 10.7.0.23-1+cuda12.6 [124 kB]
Get:12 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-headers-plugin-dev 10.7.0.23-1+cuda12.6 [6064 B]
Get:13 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-lean-dev 10.7.0.23-1+cuda12.6 [21.6 MB]
Get:14 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-plugin-dev 10.7.0.23-1+cuda12.6 [11.3 MB]
Get:15 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-vc-plugin-dev 10.7.0.23-1+cuda12.6 [80.4 kB]
Get:16 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvonnxparsers-dev 10.7.0.23-1+cuda12.6 [2147 kB]
Get:17 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  libnvinfer-samples 10.7.0.23-1+cuda12.6 [187 MB]
Get:18 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  python3-libnvinfer 10.7.0.23-1+cuda12.6 [810 kB]
Get:19 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  python3-libnvinfer-lean 10.7.0.23-1+cuda12.6 [504 kB]
Get:20 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  python3-libnvinfer-dispatch 10.7.0.23-1+cuda12.6 [504 kB]
Get:21 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  python3-libnvinfer-dev 10.7.0.23-1+cuda12.6 [2966 B]
Get:22 file:/var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6  tensorrt 10.7.0.23-1+cuda12.6 [2950 B]
Fetched 164 kB in 12s (14.0 kB/s)
Selecting previously unselected package libnvinfer10.
(Reading database ... 61399 files and directories currently installed.)
Preparing to unpack .../00-libnvinfer10_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer10 (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-lean10.
Preparing to unpack .../01-libnvinfer-lean10_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-lean10 (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-plugin10.
Preparing to unpack .../02-libnvinfer-plugin10_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-plugin10 (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-vc-plugin10.
Preparing to unpack .../03-libnvinfer-vc-plugin10_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-vc-plugin10 (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-dispatch10.
Preparing to unpack .../04-libnvinfer-dispatch10_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-dispatch10 (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvonnxparsers10.
Preparing to unpack .../05-libnvonnxparsers10_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvonnxparsers10 (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-bin.
Preparing to unpack .../06-libnvinfer-bin_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-bin (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-headers-dev.
Preparing to unpack .../07-libnvinfer-headers-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-headers-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-dev.
Preparing to unpack .../08-libnvinfer-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-dispatch-dev.
Preparing to unpack .../09-libnvinfer-dispatch-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-dispatch-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-headers-plugin-dev.
Preparing to unpack .../10-libnvinfer-headers-plugin-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-headers-plugin-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-lean-dev.
Preparing to unpack .../11-libnvinfer-lean-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-lean-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-plugin-dev.
Preparing to unpack .../12-libnvinfer-plugin-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-plugin-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvinfer-vc-plugin-dev.
Preparing to unpack .../13-libnvinfer-vc-plugin-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvinfer-vc-plugin-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package libnvonnxparsers-dev.
Preparing to unpack .../14-libnvonnxparsers-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking libnvonnxparsers-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package zlib1g-dev:amd64.
Preparing to unpack .../15-zlib1g-dev_1%3a1.2.11.dfsg-2ubuntu9.2_amd64.deb ...
Unpacking zlib1g-dev:amd64 (1:1.2.11.dfsg-2ubuntu9.2) ...
Selecting previously unselected package libnvinfer-samples.
Preparing to unpack .../16-libnvinfer-samples_10.7.0.23-1+cuda12.6_all.deb ...
Unpacking libnvinfer-samples (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package python3-libnvinfer.
Preparing to unpack .../17-python3-libnvinfer_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking python3-libnvinfer (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package python3-libnvinfer-lean.
Preparing to unpack .../18-python3-libnvinfer-lean_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking python3-libnvinfer-lean (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package python3-libnvinfer-dispatch.
Preparing to unpack .../19-python3-libnvinfer-dispatch_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking python3-libnvinfer-dispatch (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package python3-libnvinfer-dev.
Preparing to unpack .../20-python3-libnvinfer-dev_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking python3-libnvinfer-dev (10.7.0.23-1+cuda12.6) ...
Selecting previously unselected package tensorrt.
Preparing to unpack .../21-tensorrt_10.7.0.23-1+cuda12.6_amd64.deb ...
Unpacking tensorrt (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-headers-dev (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer10 (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-plugin10 (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-vc-plugin10 (10.7.0.23-1+cuda12.6) ...
Setting up libnvonnxparsers10 (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-dispatch10 (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-dispatch-dev (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-dev (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-lean10 (10.7.0.23-1+cuda12.6) ...
Setting up zlib1g-dev:amd64 (1:1.2.11.dfsg-2ubuntu9.2) ...
Setting up libnvonnxparsers-dev (10.7.0.23-1+cuda12.6) ...
Setting up python3-libnvinfer-dispatch (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-headers-plugin-dev (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-lean-dev (10.7.0.23-1+cuda12.6) ...
Setting up python3-libnvinfer (10.7.0.23-1+cuda12.6) ...
Setting up python3-libnvinfer-lean (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-plugin-dev (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-vc-plugin-dev (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-bin (10.7.0.23-1+cuda12.6) ...
Setting up libnvinfer-samples (10.7.0.23-1+cuda12.6) ...
Setting up python3-libnvinfer-dev (10.7.0.23-1+cuda12.6) ...
Setting up tensorrt (10.7.0.23-1+cuda12.6) ...
Processing triggers for libc-bin (2.35-0ubuntu3.8) ...
Processing triggers for man-db (2.10.2-1) ...
```

### Verify TensorRT Install

```zsh
> dpkg-query -W tensorrt

-----------
tensorrt        10.7.0.23-1+cuda12.6

-----------

> dpkg -l | grep nvinfer

-----------

ii  libnvinfer-bin                                     10.7.0.23-1+cuda12.6                    amd64        TensorRT binaries
ii  libnvinfer-dev                                     10.7.0.23-1+cuda12.6                    amd64        TensorRT development libraries
ii  libnvinfer-dispatch-dev                            10.7.0.23-1+cuda12.6                    amd64        TensorRT development dispatch runtime libraries
ii  libnvinfer-dispatch10                              10.7.0.23-1+cuda12.6                    amd64        TensorRT dispatch runtime library
ii  libnvinfer-headers-dev                             10.7.0.23-1+cuda12.6                    amd64        TensorRT development headers
ii  libnvinfer-headers-plugin-dev                      10.7.0.23-1+cuda12.6                    amd64        TensorRT plugin headers
ii  libnvinfer-lean-dev                                10.7.0.23-1+cuda12.6                    amd64        TensorRT lean runtime libraries
ii  libnvinfer-lean10                                  10.7.0.23-1+cuda12.6                    amd64        TensorRT lean runtime library
ii  libnvinfer-plugin-dev                              10.7.0.23-1+cuda12.6                    amd64        TensorRT plugin libraries
ii  libnvinfer-plugin10                                10.7.0.23-1+cuda12.6                    amd64        TensorRT plugin libraries
ii  libnvinfer-samples                                 10.7.0.23-1+cuda12.6                    all          TensorRT samples
ii  libnvinfer-vc-plugin-dev                           10.7.0.23-1+cuda12.6                    amd64        TensorRT vc-plugin library
ii  libnvinfer-vc-plugin10                             10.7.0.23-1+cuda12.6                    amd64        TensorRT vc-plugin library
ii  libnvinfer10                                       10.7.0.23-1+cuda12.6                    amd64        TensorRT runtime libraries
ii  python3-libnvinfer                                 10.7.0.23-1+cuda12.6                    amd64        Python 3 bindings for TensorRT standard runtime
ii  python3-libnvinfer-dev                             10.7.0.23-1+cuda12.6                    amd64        Python 3 development package for TensorRT standard runtime
ii  python3-libnvinfer-dispatch                        10.7.0.23-1+cuda12.6                    amd64        Python 3 bindings for TensorRT dispatch runtime
ii  python3-libnvinfer-lean                            10.7.0.23-1+cuda12.6                    amd64        Python 3 bindings for TensorRT lean runtime

```

To verify that your installation is working, use the following Python commands:

- Import the tensorrt Python module.
- Confirm that the correct version of TensorRT has been installed.
- Create a Builder object to verify that your CUDA installation is working.

```zsh
python -c "import tensorrt; print('TensorRT Version: ', tensorrt.__version__); import tensorrt_lean as trt; print('TensorRT Lean Version: ', trt.__version__); import tensorrt_dispatch as trt_dis; print('TensorRT Lean Version: ', trt_dis.__version__)"
```

### Installed packages

```zsh
apt list --installed | egrep cuda

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

cuda-12-6/unknown,now 12.6.1-1 amd64 [installed,upgradable to: 12.6.3-1]
cuda-cccl-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-command-line-tools-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda-compiler-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda-crt-12-6/unknown,now 12.6.85-1 amd64 [installed,automatic]
cuda-cudart-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-cudart-dev-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-cuobjdump-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-cupti-12-6/unknown,now 12.6.80-1 amd64 [installed,automatic]
cuda-cupti-dev-12-6/unknown,now 12.6.80-1 amd64 [installed,automatic]
cuda-cuxxfilt-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-demo-suite-12-6/unknown,now 12.6.68-1 amd64 [installed,upgradable to: 12.6.77-1]
cuda-documentation-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-driver-dev-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-gdb-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-keyring/unknown,now 1.1-1 all [installed]
cuda-libraries-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda-libraries-dev-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda-nsight-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-nsight-compute-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda-nsight-systems-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda-nvcc-12-6/unknown,now 12.6.85-1 amd64 [installed,automatic]
cuda-nvdisasm-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-nvml-dev-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-nvprof-12-6/unknown,now 12.6.80-1 amd64 [installed,automatic]
cuda-nvprune-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-nvrtc-12-6/unknown,now 12.6.85-1 amd64 [installed,automatic]
cuda-nvrtc-dev-12-6/unknown,now 12.6.85-1 amd64 [installed,automatic]
cuda-nvtx-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-nvvm-12-6/unknown,now 12.6.85-1 amd64 [installed,automatic]
cuda-nvvp-12-6/unknown,now 12.6.80-1 amd64 [installed,automatic]
cuda-opencl-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-opencl-dev-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-profiler-api-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-repo-wsl-ubuntu-12-6-local/now 12.6.1-1 amd64 [installed,local]
cuda-runtime-12-6/unknown,now 12.6.1-1 amd64 [installed,upgradable to: 12.6.3-1]
cuda-sanitizer-12-6/unknown,now 12.6.77-1 amd64 [installed,automatic]
cuda-toolkit-12-6-config-common/unknown,now 12.6.68-1 all [installed,upgradable to: 12.6.77-1]
cuda-toolkit-12-6/unknown,now 12.6.3-1 amd64 [installed]
cuda-toolkit-12-config-common/unknown,now 12.6.68-1 all [installed,upgradable to: 12.6.77-1]
cuda-toolkit-config-common/unknown,now 12.6.68-1 all [installed,upgradable to: 12.6.77-1]
cuda-tools-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda-visual-tools-12-6/unknown,now 12.6.3-1 amd64 [installed,automatic]
cuda/unknown,now 12.6.1-1 amd64 [installed,upgradable to: 12.6.3-1]
cudnn9-cuda-12-6/unknown,now 9.6.0.74-1 amd64 [installed,automatic]
cudnn9-cuda-12/unknown,now 9.6.0.74-1 amd64 [installed,automatic]
libcudart11.0/jammy,now 11.5.117~11.5.1-1ubuntu1 amd64 [installed,automatic]
libcudnn9-cuda-12/unknown,now 9.6.0.74-1 amd64 [installed,automatic]
libcudnn9-dev-cuda-12/unknown,now 9.6.0.74-1 amd64 [installed,automatic]
libcudnn9-static-cuda-12/unknown,now 9.6.0.74-1 amd64 [installed,automatic]
nvidia-cuda-dev/jammy,now 11.5.1-1ubuntu1 amd64 [installed,automatic]
nvidia-cuda-gdb/jammy,now 11.5.114~11.5.1-1ubuntu1 amd64 [installed,automatic]
nvidia-cuda-toolkit-doc/jammy,now 11.5.1-1ubuntu1 all [installed,automatic]
nvidia-cuda-toolkit/jammy,now 11.5.1-1ubuntu1 amd64 [installed]
```
