FROM python:3.8-slim-buster

# AWS CLI and update Packages
RUN apt update -y && apt install awscli -y

# Setting the working directory
WORKDIR /app

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Copying only the requirements file first for catching purpose
COPY requirements.txt /app/requirements.txt

# installing all the requirements
RUN pip install -r requirements.txt

# copying rest of the application files
COPY . /app

# Command to run the application
CMD ["python3","app.py"]