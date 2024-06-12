FROM quay.io/centos/centos:stream9

COPY Nvidia.repo /etc/yum.repos.d/Nvidia.repo
RUN dnf update
RUN dnf install cuda-toolkit-11-8-11.8.0-1.x86_64 -y
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN dnf install git make -y
RUN git clone https://github.com/NVIDIA/cuda-samples.git
RUN cd cuda-samples && git checkout tags/v11.8
RUN cd cuda-samples/Samples/1_Utilities/deviceQuery && make
CMD ["cuda-samples/Samples/1_Utilities/deviceQuery/deviceQuery"]
