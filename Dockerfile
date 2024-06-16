FROM quay.io/centos/centos:stream9

COPY Nvidia.repo /etc/yum.repos.d/Nvidia.repo
COPY cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
COPY app.py /usr/share/applications/app.py

RUN dnf update -y
RUN dnf install cuda-toolkit-12-5 -y
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN dnf install xz -y
RUN cd /tmp && tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
RUN cp /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda/include 
RUN cp -P /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64 
RUN chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
RUN curl --output ~/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir ~/miniconda
RUN chmod -R 770 ~/miniconda
RUN chmod +x ~/Miniconda3-latest-Linux-x86_64.sh
RUN ~/Miniconda3-latest-Linux-x86_64.sh -b -f -p ~/miniconda
ENV PATH=/root/miniconda/bin:$PATH
RUN conda create -n tf python=3.11 pip -y
RUN conda init && source ~/.bashrc && conda activate tf
RUN pip install tensorflow==2.16.1
CMD ["python", "/usr/share/applications/app.py"]
