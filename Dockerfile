FROM tensorflow/tensorflow:latest-py3
WORKDIR /mnist
COPY . /mnist
RUN pip install --trusted-host pypi.mirrors.ustc.edu.cn/simple/ -r requirements.txt
EXPOSE 80
ENV NAME World
CMD ["python", "mnist.py"]
