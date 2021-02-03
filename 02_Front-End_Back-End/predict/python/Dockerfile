FROM python:3.6
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && \
	rm /tmp/requirements.txt
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html