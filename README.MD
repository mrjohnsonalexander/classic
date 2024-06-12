# Introduction
Enable workloads on Nvidia GPU.
## CentOS Stream 9
### Distribution
#### ISO
https://mirrors.centos.org/mirrorlist?path=/9-stream/BaseOS/x86_64/iso/CentOS-Stream-9-latest-x86_64-dvd1.iso
##### K8 API
```Bash
git clone https://oauth2:<PAT>@github.com/mrjohnsonalexander/classic.git
cd classic
sudo dnf install ansible-core
ansible-galaxy collection install community.general
sudo dnf install python-jmespath
sudo ansible-playbook -i ansible/development ansible/site.yml --connection=local
sudo kubectl get pods --all-namespaces
sudo kubeadm token create --print-join-command
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
curl --output /etc/yum.repos.d/Centos-Project.repo https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/Centos-Project.repo
curl --output /etc/yum.repos.d/Nvidia.repo https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/Nvidia.repo
curl --output /etc/yum.repos.d/nvidia-container-toolkit.repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
curl --output /etc/yum.repos.d/docker-ce.repo https://download.docker.com/linux/centos/docker-ce.repo
dnf update
```
###### Build Cuda Samples in Centos Stream 9 container, run using systemd on WSL2, and then push to registry.
```Bash
curl --output /etc/wsl.conf https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/wsl.conf
exit
wsl --shutdown
wsl -d CentOS
dnf install nvidia-container-toolkit
dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker buildx build -f Dockerfile -t localhost:5000/cuda .
docker run --rm -it --gpus=all localhost:5000/cuda
/usr/libexec/docker/cli-plugins/docker-compose up -d
docker push localhost:5000/cuda
```
##### Enable WSL to connect to Hyper-VM
```PowerShell
Get-NetIPInterface | where {$_.InterfaceAlias -eq 'vEthernet (WSL)' -or $_.InterfaceAlias -eq 'vEthernet (External)'} | Set-NetIPInterface -Forwarding Enabled
```
###### Join WSL to Hyper-V K8 API
```Bash
cp ./ansible/roles/k8controlplane/files/config.toml /etc/containerd/config.toml
cat > /etc.yum.repos.d/kubernetes.repo <<EOL
[kubernetes]
name=kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.30/rpm/
enabled=1
gpgcheck=0
EOL
dnf install kubectl kubeadm kubelet
systemctl restart containerd
systemctl start docker
kubeadm join <IP ADDRESS:PORT> --token <TOKEN> --discovery-token-ca-cert-has <HASH>
```
```
### Notes
#### Drivers
```Bash
nvidia-smi
```
