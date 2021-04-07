FROM ubuntu:latest

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get install -y git wget sudo curl cmake python3 python3-dev python3-pip ffmpeg libopencv-dev python3-opencv jupyter

RUN pip3 install numpy pandas matplotlib scikit-learn jupyterlab pillow torch torchvision jupyter_client

RUN pip3 install tensorflow
RUN pip3 install fancyimpute
RUN pip3 install graphviz
RUN apt-get install -y graphviz

RUN groupadd -g 999 user && useradd -r -u 999 -g user -ms /bin/bash user && usermod -aG sudo user && usermod -u 1000 user
RUN echo "\nuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN usermod -a -G video user

USER user
WORKDIR /home/user
CMD ["jupyter-lab",  "--ip='*'", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

