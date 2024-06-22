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
RUN python3.11 -m pip install tensorrt-libs==8.6.1 --extra-index-url https://pypi.nvidia.com --no-dependencies
RUN python3.11 -m pip install tensorrt-bindings==8.6.1 --extra-index-url https://pypi.nvidia.com --no-dependencies
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
RUN ln -s /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_plugin.so.8 /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_plugin.so.8.6.1
RUN ln -s /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer.so.8 /usr/local/lib/python3.11/site-packages/tensorrt_libs/libnvinfer.so.8.6.1
RUN python3.11 -m pip install pip==21.3.1
COPY requirements.txt /usr/share/applications/requirements.txt
RUN python3.11 -m pip install -r /usr/share/applications/requirements.txt
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ENV KAGGLE_KEY=$KAGGLE_KEY
RUN python3.11 -c "import kagglehub; kagglehub.model_download('keras/gemma/keras/gemma_1.1_instruct_2b_en')"
RUN python3.11 -m pip install python-multipart
COPY templates/index.html /usr/share/applications/templates/index.html
EXPOSE 4000
COPY main.py /usr/share/applications/main.py
CMD ["python3.11", "-m", "uvicorn", "--app-dir", "/usr/share/applications", "main:app", "--reload", "--host", "0.0.0.0", "--port", "4000"]