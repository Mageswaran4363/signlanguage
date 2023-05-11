import tensorflow as tf
from tensorflow import keras
import numpy as np

#Loading themodel
batch_size = 32
img_height = 64
img_width = 64
model_dl = keras.models.load_model("model_dl.h5") #look for local saved file

from keras.preprocessing import image

#Creating a dictionary to map each of the indexes to the corresponding number or letter

dict = {6:"call",0:"Doctor",1:"Help",2:"Hot",3:"Lose",4:"Pain",5:"Theif"}

#Predicting images

from tkinter.filedialog import askopenfile

file = askopenfile(filetypes =[('file selector', '*.jpeg')])
print(str(file.name))
img = image.load_img(str(file.name), target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

image = np.vstack([x])
classes = model_dl.predict_classes(image, batch_size=batch_size)
probabilities = model_dl.predict_proba(image, batch_size=batch_size)
probabilities_formatted = list(map("{:.2f}%".format, probabilities[0]*100))

import ctypes  # An included library with Python install.
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
Mbox('sign is', str(dict[classes.item()]), 1)

print(f'The predicted : "{dict[classes.item()]}"')

