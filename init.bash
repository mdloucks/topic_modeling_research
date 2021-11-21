#!bin/bash

# init script to create venv and install dependencies
sudo apt-get update
sudo apt install python3-pip

# create virtual environment
python3 -m venv venv

source venv/bin/activate

pip install numpy
pip install sklearn
pip install matplotlib
pip install nltk
pip install networkx
pip install pandas
pip install stop_words