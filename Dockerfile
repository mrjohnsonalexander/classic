FROM quay.io/centos/centos:stream9

COPY ./configs/Nvidia.repo /etc/yum.repos.d/Nvidia.repo
RUN dnf update -y
RUN dnf install libnccl libnccl-devel libnccl-static -y
RUN dnf install python3.11 -y
RUN python3.11 -m ensurepip
RUN python3.11 -m pip install pip==21.3.1
COPY requirements.txt /usr/share/applications/requirements.txt
RUN python3.11 -m pip install -r /usr/share/applications/requirements.txt
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ENV KAGGLE_KEY=$KAGGLE_KEY
RUN python3.11 -c "import kagglehub; kagglehub.model_download('google/gemma/transformers/1.1-2b-it/1')"
COPY templates/index.html /usr/share/applications/templates/index.html
EXPOSE 4000
COPY main.py /usr/share/applications/main.py
CMD ["python3.11", "-m", "uvicorn", "--app-dir", "/usr/share/applications", "main:app", "--reload", "--host", "0.0.0.0", "--port", "4000"]