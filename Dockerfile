FROM quay.io/centos/centos:stream9

COPY Nvidia.repo /etc/yum.repos.d/Nvidia.repo
COPY cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
RUN dnf update -y
RUN dnf install cuda-toolkit-12-5 -y
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN dnf install xz -y
RUN cd /tmp && tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
RUN cp /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda/include 
RUN cp -P /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64 
RUN chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
RUN rm -rf /tmp/*
RUN dnf install python3.11 -y
RUN python3.11 -m ensurepip
RUN python3.11 -m pip install pip==24.0
RUN dnf install procps -y
RUN python3.11 -m pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com --no-dependencies
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
RUN python3.11 -m pip install pip==21.3.1
RUN python3.11 -m pip install tensorflow[and-cuda]==2.16.1 --no-dependencies
COPY requirements.txt /usr/share/applications/requirements.txt
RUN python3.11 -m pip install -r /usr/share/applications/requirements.txt
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ENV KAGGLE_KEY=$KAGGLE_KEY
COPY main.py /usr/share/applications/main.py
CMD ["python3.11", "/usr/share/applications/main.py"]
