# Introduction
Enable workloads on Nvidia GPU.
## CentOS Stream 9
### Distribution 
#### WSL2
```PowerShell
git clone https://github.com/CentOS/sig-cloud-instance-images.git
cd sig-cloud-instance-images
git switch CentOS-Stream-9-x86_64
wsl --import CentOS .\sig-cloud-instance-images .\sig-cloud-instance-images\docker\centos-stream -9-x86_64.tar.xz
wsl -d CentOS
```
### Packages
#### RPM
##### Repositorys
```Bash
mkdir /etc/yum.repos.d
curl --output /etc/yum.repos.d/Centos-Project.repo https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/Centos-Project.repo
curl --output /etc/yum.repos.d/Nvidia.repo https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/Nvidia.repo
curl --output /etc/yum.repos.d/nvidia-container-toolkit.repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
curl --output /etc/yum.repos.d/docker-ce.repo https://download.docker.com/linux/centos/docker-ce.repo
dnf update
```
##### Cuda 11-8
``` Bash
dnf install cuda-toolkit-11-8-11.8.0-1.x86_64
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
nvcc --version
```
###### Cuda Samples
```Bash
dnf install git
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
git checkout tags/v11.8
cd Samples/1_Utilities/deviceQuery
make
./deviceQuery
```
##### Run Nvidia Containers using systemd on WSL2
```Bash
curl --output /etc/wsl.conf https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/wsl.conf
exit
wsl --shutdown
wsl -d CentOS
dnf install nvidia-container-toolkit
dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2 nbody -gpu -benchmark
```
#### Conda
```bash
curl --output Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh
```
##### Tensorflow-GPU 1.12
``` Bash
conda create -n tf python=3.6 tensorflow-gpu=1.12 cudatoolkit=9.0
conda activate tf
export LD_LIBRARY_PATH=/root/miniconda3/pkgs/cudatoolkit-9.0-h13b8566_0/lib
python -c "import tensorflow as tf; print(tf.test.gpu_device_name())"
```
##### PyTorch 1.13
``` Bash
conda create -n pt python=3.8
conda activate pt
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
### Notes
#### Drivers
```Bash
nvidia-smi
```
