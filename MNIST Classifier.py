#!/usr/bin/env python
# coding: utf-8

# ## MNIST

# In[3]:


#importing tensorflow 
import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/mnist.npz"


# In[4]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[12]:


def train_mnist_conv():
    
    #defing the class for callback initialisation
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    
    
    mnist = tf.keras.datasets.mnist
    
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
   
    training_images=training_images.reshape(60000, 28, 28, 1)    #reshaping the traiaing set
    training_images=training_images/255.0                        #normalising the training
    test_images=test_images.reshape(10000, 28, 28, 1)    #reshaping the test data
    test_images=test_images/255.0                                #normalising the test images 
    
    #Creating an object for the class
    callbacks=myCallback()     
    
    #Building modeL
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation= 'relu',input_shape=(28,28,1)),tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Flatten(),tf.keras.layers.Dense(512,activation = 'relu'),tf.keras.layers.Dense(10,activation = 'softmax')])
    
    #Compilig the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    # model fitting
    history = model.fit(training_images,training_labels,epochs= 10,callbacks=[callbacks])
    
    loss,accuracy = model.evaluate(test_images,test_labels)
    print("Model acc for test images is",accuracy*100,"%.")
    

    return history.epoch, history.history['acc'][-1]


# In[13]:


#Running the function
train_mnist_conv()


# In[10]:


#Saving the notebook


# In[15]:


get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

