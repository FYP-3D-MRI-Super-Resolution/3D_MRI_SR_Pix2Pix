FROM python:3.8

ADD . .

# Fix for running cv2
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "train.py", \
"--dataset_mode", "nifti", \
"--model", "pix2pix3d", \
"--name", "t1t2", \
"--n_epochs", "200", \
"--n_epochs_decay", "200", \
"--suffix", "{model}_{batch_size}_{lambda_L1}_{lambda_L2_T}" \
]