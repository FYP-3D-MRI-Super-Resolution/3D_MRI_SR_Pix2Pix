FROM nvcr.io/nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

# Fix for running cv2: ffmpeg libsm6 libxext6
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install python3-dev python3-pip ffmpeg libsm6 libxext6 -y

WORKDIR Pix2PixNIfTI
ADD . .

RUN pip3 install -r requirements.txt

CMD ["bash", "./scripts/brain_pix2pix/grid_search.sh"]