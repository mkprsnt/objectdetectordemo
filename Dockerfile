# Start from anaconda image 
FROM continuumio/anaconda3

# Create working directory and copy the project contents into it
COPY . /usr/app/
WORKDIR /usr/app/

# Install packages required for the app
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip3 install -r requirements.txt

# Initiate the app
EXPOSE 8000
CMD python3 objDetector.py