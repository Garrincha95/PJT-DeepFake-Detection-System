FROM tensorflow/tensorflow:1.15.2-gpu-py3
COPY requirements.txt /tmp/
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install -r /tmp/requirements.txt && \
	rm /tmp/requirements.txt
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html