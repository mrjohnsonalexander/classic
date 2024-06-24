# Introduction
Enable Large Language Model (LLM) solution.
# Background
This software builds a LLM system enabled for Nvidia GPU, by running a container on WSL2 Centos distribution, using a Python implementation.
## CentOS Stream 9
### Distribution
#### Hyper V VM Centos
Create Gen 2 Hyper VM using [Centos ISO](https://mirror.stream.centos.org/9-stream/BaseOS/x86_64/iso/CentOS-Stream-9-20240617.0-x86_64-boot.iso) with at least 2 Virtual Cores.
##### K8 API
```Bash
sudo dnf install git-core -y
git clone git clone https://oauth2:<PAT>@github.com/mrjohnsonalexander/classic.git
sudo dnf install ansible-core python-jmespath -y
sudo ansible-galaxy collection install community.general
sudo ansible-playbook -i ansible/development ansible/site.yml --connection=local
sudo kubectl get pods --all-namespaces
```
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
curl --output /etc/yum.repos.d/Centos-Project.repo https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/configs/Centos-Project.repo
curl --output /etc/yum.repos.d/Nvidia.repo https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/configs/Nvidia.repo
curl --output /etc/yum.repos.d/nvidia-container-toolkit.repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
curl --output /etc/yum.repos.d/docker-ce.repo https://download.docker.com/linux/centos/docker-ce.repo
dnf update
```
###### Install cuda dependent Tensorflow in Centos Stream 9 container, run using systemd on WSL2, and then push to registry.
```Bash
curl --output /etc/wsl.conf https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/configs/wsl.conf
exit
wsl --shutdown
wsl -d CentOS
dnf install nvidia-container-toolkit
dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
cp ./classic/configs/config.toml /etc/containerd/config.toml
systemctl restart docker
docker buildx build -f Dockerfile -t localhost:5000/cuda . --build-arg KAGGLE_USERNAME=<USER NAME> --build-arg KAGGLE_KEY=<KEY>
docker run -d -p 4000:4000 --gpus=all localhost:5000/cuda
curl localhost:4000/health
curl localhost:4000/generate
/usr/libexec/docker/cli-plugins/docker-compose up -d
docker push localhost:5000/cuda
```
### Kustomize Apply to K8 cluster
```Bash
kubectl apply -k kustomize
```
### Notes
#### Form
localhost:4000/form
#### Drivers
```Bash
nvidia-smi
```
#### Docker registry
```Bash
cp classic/daemon.json /etc/docker/daemon.json
```
#### CentOS Stream 9 ISO for Hyper VM K8 Api to WSL Kubelet forwarding
```PowerShell
 Get-NetIPInterface | where {$_.InterfaceAlias -eq 'vEthernet (WSL)' -or $_.InterfaceAlias -eq 'vEthernet (External)'} | Set-NetIPInterface -Forwarding Enabled -Verbose
```
#### Stack
- Python Version 3.11
- Docker Community Version 26.1.4
- Kubernetes v1.30
- Containerd  1.6.33
- WSL Distribution Centos Stream9
- WSL version: 2.2.4.0
- OS Version Windows 10 BUILD 19045
- Nvidia Game Ready Driver Version: 555.99 
- Installed Physical Memory (RAM) 32 GB
- Nvidia Geforce RTX 4060 Ti 8 GB VRAM
- Intel CPU i7-4820k