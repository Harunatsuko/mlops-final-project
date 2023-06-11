# pull official base image
FROM ubuntu:latest

# set work directory
WORKDIR /usr/src/app

# copy requirements file
COPY ./requirements.txt /usr/src/app/requirements.txt

# install python
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y curl python3 python3-pip

# install dependencies
RUN pip install -r /usr/src/app/requirements.txt

# copy project
COPY . /usr/src/app/
