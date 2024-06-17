FROM quay.io/centos/centos:stream9

COPY Nvidia.repo /etc/yum.repos.d/Nvidia.repo
COPY cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
COPY main.py /usr/share/applications/main.py

RUN dnf update -y
RUN dnf install cuda-toolkit-12-5 -y
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN dnf install xz -y
RUN cd /tmp && tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
RUN cp /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h /usr/local/cuda/include 
RUN cp -P /tmp/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64 
RUN chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
RUN dnf install python3.11 -y
RUN python3.11 -m ensurepip
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install --upgrade keras-nlp
RUN python3.11 -m pip install --upgrade keras>=3
RUN python3.11 -m pip install kaggle
RUN python3.11 -m pip install numpy
RUN python3.11 -m pip install tensorrt
RUN python3.11 -m pip install tensorrt==8.6.1
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
RUN ln -s /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_plugin.so.8 /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_plugin.so.8.6.1
RUN ln -s /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer.so.8 /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer.so.8.6.1
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ENV KAGGLE_KEY=$KAGGLE_KEY
CMD ["python3.11", "/usr/share/applications/main.py"]
