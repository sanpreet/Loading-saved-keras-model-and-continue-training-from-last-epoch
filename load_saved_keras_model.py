""" Loading saved keras model and continue training from last epoch"""
# Importing the required libraries
import os
import glob
from numpy import loadtxt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input predictors (X) and output (y) variable
X = dataset[:,0:8]
y = dataset[:,8]

# keras model is saved usinf ModelCheckpoint
# weight is the name of folder
# model is saved with the name 'weights.{epoch:02d}.{accuracy:.4f}.hdf5
# model is saved if the new accuracy is better than previous one
ckpt_callback = ModelCheckpoint(filepath=os.path.join('weight', 'weights.{epoch:02d}.{accuracy:.4f}.hdf5'), monitor = 'accuracy', save_best_only=True)

# creating the keras sequential model
model = Sequential()
# fully connected layer with 12 neurons is created
# input dimension shape must be equal to number of features of the dataset
# last layer has one neuron as this is binary classifier and there would be one answer/label with respect to real time predictor 
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model is compiled with loss as binary cross entropy, optimizer as adam and  metrics as acuracy 
# accuracy is chosen as metrics because this is a classification problem.
model.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# this is the directory where weights are saved
checkpoint_path = 'weight'

# this is function to load weight file for resuming training from last epoch
def checkpoint_function():
    # if there is no file being saved in the checkpoint path then routine will go to loop model.fit(X, y, epochs=150, batch_size=1, callbacks=[ckpt_callback])
    # this signifies that no file is saved in the checkpoint path and let us begin the training from the first epoch.
    if not os.listdir(checkpoint_path):
        return
    # this loop will fetch epochs number in a list    
    files_int = list()
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('.')[1])
        files_int.append(epoch)
    # getting the maximum value for an epoch from the list
    # this is reference value and will help to find the file with that value    
    max_value = max(files_int)
    # conditions are applied to find the file which has the maximum value of epoch
    # such file would be the last file where the training is stopped and we would like to resume the training from that point.
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('.')[1])
        if epoch > max_value:
            pass
        elif epoch < max_value:
            pass
        else:
            final_file = i    
    # in the end we will get the epoch as well as file         
    return final_file, max_value        
            
checkpoint_path_file = checkpoint_function()
# this is interesting loop of the code. Code has been divided into conditions here
# in the if condition if checkpoint is not none, then file and last epcoh is restored
# Such file is loaded using load_model and training start from that epoch
if checkpoint_path_file is not None:
     # Load model:
    checkpoint_path_file = checkpoint_function()[0]
    max_value = checkpoint_function()[1]

    model = load_model(os.path.join(checkpoint_path, checkpoint_path_file))
    model.fit(X, y, epochs=150, batch_size=1, callbacks=[ckpt_callback], initial_epoch = max_value)    
# if there is no checkpoint_path_file it means no file is saved and hence model will be trained from scratch
# file is saved in form of .hdf5 extension 
else:
    model.fit(X, y, epochs=150, batch_size=1, callbacks=[ckpt_callback])
