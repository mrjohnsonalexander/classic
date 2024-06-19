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
##### K8 API
```Bash
git clone https://oauth2:<PAT>@github.com/mrjohnsonalexander/classic.git
cd classic
sudo dnf install ansible-core
ansible-galaxy collection install community.general
sudo dnf install python-jmespath
sudo ansible-playbook -i ansible/development ansible/site.yml --connection=local
sudo kubectl get pods --all-namespaces
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
###### Install cuda dependent Tensorflow in Centos Stream 9 container, run using systemd on WSL2, and then push to registry.
```Bash
curl --output /etc/wsl.conf https://raw.githubusercontent.com/mrjohnsonalexander/classic/main/wsl.conf
exit
wsl --shutdown
wsl -d CentOS
dnf install nvidia-container-toolkit
dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
nvidia-ctk runtime configure --runtime=docker
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
#### Drivers
```Bash
nvidia-smi
```
#### Cuddn
```Bash
mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz ../classic/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```
#### Docker registry
```Bash
cp classic/daemon.json /etc/docker/daemon.json
```

