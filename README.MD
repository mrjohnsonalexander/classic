# Introduction
Enable Large Language Model (LLM) solution.
# Background
This software builds a LLM system enabled for Nvidia GPU, by running a container on WSL2 Centos distribution, using a Python implementation. Running using Docker is working. Deploying built container to K8 kubelet on WSL2 is working, however haven't reproduced [device plugin](https://github.com/NVIDIA/k8s-device-plugin/issues/332) enabling kubelet managed container on WSL2 to use GPU device; so submitting form and invoking generate with managed by kublet is not working yet.
## CentOS Stream 9
### Distribution
#### Hyper V VM Centos
Create Gen 2 Hyper VM using [Centos ISO](https://mirror.stream.centos.org/9-stream/BaseOS/x86_64/iso/CentOS-Stream-9-20240617.0-x86_64-boot.iso) with at least 2 Virtual Cores.
##### Hyper-V VM hosted K8 API
```Bash
sudo dnf install git-core -y
git clone git clone https://oauth2:<PAT>@github.com/mrjohnsonalexander/classic.git
sudo dnf install ansible-core python-jmespath -y
sudo ansible-galaxy collection install community.general
sudo ansible-playbook -i ansible/development ansible/site.yml --connection=local
sudo kubectl get pods --all-namespaces
```
#### WSL2 Centos Distribution
```PowerShell
git clone https://github.com/CentOS/sig-cloud-instance-images.git
cd sig-cloud-instance-images
git switch CentOS-Stream-9-x86_64
wsl --import CentOS .\sig-cloud-instance-images .\sig-cloud-instance-images\docker\centos-stream -9-x86_64.tar.xz
wsl -d CentOS
```
##### Configure Systemd and Package repo for WSL CentOS distribution
```Bash
mkdir /etc/yum.repos.d
curl --output /etc/yum.repos.d/Centos-Project.repo https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/configs/Centos-Project.repo
dnf update -y
dnf install git-core -y
git clone  https://github.com/mrjohnsonalexander/classic.git
cp ./classic/configs/wsl.conf /etc/wsl.conf
exit
```
###### Configure WSL distribution as Nvidia Container host
```PowerShell
wsl --shutdown
wsl -d CentOS
```
```Bash
dnf install ansible-core python-jmespath -y
ansible-galaxy collection install community.general
ansible-playbook -i ansible/development ansible/gpuworker.yml --connection=local
```
###### Join GPU enabled WSL Kublet to K8 API
####### Enable network forwarding between HyperV VM and WSL distributions
```PowerShell
 Get-NetIPInterface | where {$_.InterfaceAlias -eq 'vEthernet (WSL)' -or $_.InterfaceAlias -eq 'vEthernet (External)'} | Set-NetIPInterface -Forwarding Enabled -Verbose
```
####### From Hyper VM Get Join Comand
```Bash
sudo kubeadm token create --print-join-command
```
####### Join WSL distribution to Hyper V hosted K8 API
```Bash
kubeadm join <IP:PORT> --token <TOKEN> --discovery-token-ca-cert-hash <HASH>
```
##### Build and run Cuda dependent container on WSL2
###### Build and push container to registry on Hyper-VM that host K8 API
``` Bash
sudo firewall-cmd --zone=public --add-port=5000/tcp --permanent
sudo systemctl restart firewalld
sudo /usr/libexec/docker/cli-plugins/docker-compose -f ./classic/configs/docker-compose.yaml up -d
cd classic
sudo docker buildx build -f Dockerfile -t localhost:5000/cuda . --progress=plain --build-arg KAGGLE_USERNAME=<USERNAME> --build-arg KAGGLE_KEY=<KEY>
cp configs/daemon.json /etc/docker/daemon.json
sudo systemctl restart docker
sudo docker push localhost:5000/cuda
```
###### From WSL enable containerd registery 
``` Bash
vi /etc/containerd/config.toml
...
  endpoint = ["http://localhost:5000"]
...
  endpoint = ["http://<IP ADDRESS>:5000"]
:wq!
... 
systemctl restart containerd
```
###### From Hyper VM hosting K8 API apply Kustomize
```Bash
sudo kubectl label nodes <NODE NAME> nodetype=gpu
sudo kubectl apply -k kustomize
sudo kubectl get pods -n gpu
sudo kubectl get svc -n gpu
```
### Notes
#### Form
- <WSL IP ADDR>:4000/form
- <WSL IP ADDR>:4000/health
- <WSL IP ADDR>:4000/generate
#### Drivers
```Bash
nvidia-smi
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