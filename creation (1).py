import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import csv
import datetime
import time
from tensorflow.keras.models import load_model
from keras.preprocessing import image                  
from scipy.spatial import distance as dist
import argparse
import imutils
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

##year = input("enter the year")
##print("year:" +year+ ".")
##
##dept = input("enter the department")
##print("dept:" + dept + ".")
##
##sem = input("enter the semester")
##print("sem:" + sem + ".")
##
##subj = input("enter the subject")
##print("subject:" + subj + ".")

name = input("enter the name")
print("name:" + name + ".")

 # Replace with your desired folder name
path = "C:/Users/ak398/OneDrive/Desktop/face project/data"+"/"+name  # Replace with the directory path where you want to create the folder

if not os.path.exists(path):
    os.makedirs(path)
    print("Folder created successfully.")
else:
    print("Folder already exists.")


# Define the path where you want to store the captured images
path = path

# Define the number of images to capture
num_images = 40




