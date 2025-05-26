#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading the required packages


# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam


# In[3]:


#Example to load an image from the location specified.


# In[4]:


img=image.load_img(r"C:\Users\KISHORE\Desktop\training set\0-2\169.png")


# In[5]:


plt.imshow(img)


# In[6]:


#Reading an image in the pixel format into an array 


# In[7]:


cv2.imread(r"C:\Users\KISHORE\Desktop\train_img\0-2\86709.jpg")


# In[8]:


cv2.imread(r"C:\Users\KISHORE\Desktop\training set\0-2\169.png").shape #Obtaining the dimensions of an image in RGB.


# In[9]:


train=ImageDataGenerator(rescale=1/255)


# In[10]:


train_dataset = train.flow_from_directory("C:/Users/KISHORE/Desktop/training set/"                                                       
                                    ,target_size=(200,200),
                                       batch_size=3,
                                       class_mode='categorical')


# In[11]:


#Exploratory Data Analysis(EDA)


# In[12]:


# Plotting the balance of images in different age-ranges in a barplot.
import seaborn as sns
l=[]
dir_path="C:/Users/KISHORE/Desktop/training set/"  
for i in os.listdir(dir_path):
    a=[]
    for j in list(set(os.listdir(dir_path+"//"+i))- {'desktop.ini'}):
        img=image.load_img(dir_path+"//"+i+"//"+j,target_size=(200,200))
        a.append(j)
    l.append(a)
print(len(l))
x=[]
for i in l:
    x.append(len(i))
print(x)
plt.figure(figsize=(12, 8))

ax = sns.barplot(x=['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70'],
                 y=x, color='royalblue')
ax.tick_params(axis='both', labelsize=12)

plt.xlabel("Age-ranges (classes)", fontsize=14)
plt.ylabel("No. of Images", fontsize=14)

plt.title("Barplot showing balance of images in different\nAge-ranges (classes) for Age Classifier dataset", fontsize=18)


# In[13]:


train_dataset.class_indices


# In[14]:


train_dataset.classes


# In[15]:


model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(200,200,3)),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                 tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                 tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  tf.keras.layers.Dense(9,activation='softmax')
                                 ])


# In[16]:


model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


# In[17]:


model.summary()


# In[18]:


model_fit=model.fit(train_dataset,epochs=10)


# In[19]:


dir_path="C:/Users/KISHORE/Desktop/test_image/"
for i in os.listdir(dir_path):
    for j in list(set(os.listdir(dir_path+"//"+i))- {'desktop.ini'}):
        val=[]
        img=image.load_img(dir_path+"//"+i+"//"+j,target_size=(200,200))
        plt.imshow(img)
        plt.show()
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (200, 200))
        img = tf.reshape(img, (-1, 200, 200, 3))
        vals = model.predict(img/255)
        val.append(np.argmax(vals))
        val=val[0]
        if val==0:
            print('0-2')
        elif val==1:
            print('10-19')
        elif val==2:
            print('20-29')
        elif val==3:
            print('3-9')
        elif val==4:
            print('30-49')
        elif val==5:
            print('40-49')
        elif val==6:
            print('50-59')
        elif val==7:
            print('60-69')
        elif val==8:
            print('more than 70')


# In[20]:


predictions = []
l=[]
for i in os.listdir(dir_path):
    for j in list(set(os.listdir(dir_path+"//"+i))- {'desktop.ini'}):
        img=image.load_img(dir_path+"//"+i+"//"+j,target_size=(200,200))
        l.append(j)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (200, 200))
        img = tf.reshape(img, (-1, 200, 200, 3))
        prediction = model.predict(img/255)
        predictions.append(np.argmax(prediction))
import pandas as pd
my_submission = pd.DataFrame({'image_id':l , 'label': predictions})
#my_submission.to_csv('submission.csv', index=False)

# Submission file output
print("Submission File: \n---------------\n")
print(my_submission) # Displaying first five predicted output


# In[21]:



test_dataset = "C:/Users/KISHORE/Desktop/test_image/"          
test=ImageDataGenerator(rescale=1/255)
test_set = test.flow_from_directory(test_dataset,target_size=(200,200),
                                       batch_size=3,
                                       class_mode='categorical')                                 
                                       
model.evaluate(test_set,batch_size=2)


# In[ ]:


import tensorflow as tf
labels =  {0:"0-2", 1: "3-9" , 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7:"60-69",8:"more than 70"} 

def classify_image(inp):
    inp = inp.reshape(-1, 200, 200, 3)
    prediction=model.predict(inp)[0]
    confidences = {labels[i]: float(prediction[i]) for i in range(9)}
    return confidences
import gradio as gr

gr.Interface(fn=classify_image, 
             inputs=gr.Image(shape=(200, 200)),
             outputs=gr.Label(num_top_classes=3),).launch(debug='True')

