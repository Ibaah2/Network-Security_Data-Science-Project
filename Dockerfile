# We will wirte the docker file here
# These are a set of instructions to build an image

# Use a lightweight Python base image
FROM python:3.10-slim-buster

# Set working directory /app is just a name for working directory inside this container
# The /app will be created as a folder in docker
WORKDIR /app 

# Copy ALL files 
# We copy all the files under 'NETWORK_SECURITY' project into the /app folder in docker
COPY . /app

RUN apt update -y && apt install awscli -y

# Install dependencies
RUN apt-get update && pip install -r requirements.txt

# Command to run when the container starts
CMD ["python3", "app.py"]