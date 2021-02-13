FROM nvcr.io/nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

ADD . .

# Fix for running cv2: ffmpeg libsm6 libxext6
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install python3-dev python3-pip ffmpeg libsm6 libxext6 -y

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "train.py", \
"--dataset_mode", "nifti", \
"--model", "pix2pix3d", \
"--name", "t1t2", \
"--n_epochs", "200", \
"--n_epochs_decay", "200", \
"--suffix", "{model}_{batch_size}_{lambda_L1}_{lambda_L2_T}" \
]