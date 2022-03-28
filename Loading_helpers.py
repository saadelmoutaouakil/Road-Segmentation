import os
import matplotlib.image as mpimg
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as transformation
import torchvision.transforms as tvf
import matplotlib.pyplot as plt
from PIL import Image


'''
    This module contains the functions to load and transform images
'''

IMG_PATCH_SIZE = 16
TRAINING_SIZE = 100
VALIDATION_RATIO = 0.30
THRESHOLD_ROAD_BGRD_LABELS = 0.50

REL_PATH = os.path.abspath(os.curdir)
TRAINING_IMAGE_PATH = REL_PATH + "/training/images/"
TRAINING_GROUNDTRUTH_PATH = REL_PATH + "/training/groundtruth/"


def load(split=True):
    '''load images from `TRAINING_IMAGE_PATH` and its corresponding labels, split them into training and validation set
    split: if True split into train and validation set otherwise returns the whole data-set
    returns:
        train and validation set with its corresponding labels

    '''	
    training = []
    labels = []
    filename = TRAINING_IMAGE_PATH
    for i in range(1, TRAINING_SIZE+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"

        print('Loading ' + image_filename)
        img = Image.open(image_filename)
        training.append(img)


    filename = TRAINING_GROUNDTRUTH_PATH
    for i in range(1, TRAINING_SIZE+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"

        print('Loading ' + image_filename)
        img = Image.open(image_filename)
        labels.append(img)
    if(split):
        return train_test_split(training,labels,test_size=VALIDATION_RATIO)
    else :
        return training,None,labels,None
   



class Image_Dataset(Dataset):

    def __init__(self, train_data, label_data,transformation=False,normalize=None):
        self.normalize = normalize
        if(transformation):
            print('Transformations started..')
            self.train_data, self.label_data = self.apply_transformations(train_data,label_data)
            
            self.label_data = (self.label_data > THRESHOLD_ROAD_BGRD_LABELS)*1.0
            self.train_data = torch.from_numpy(self.train_data).float()
            self.train_data = torch.permute(self.train_data,(0,3,1,2))
            self.label_data = torch.from_numpy(self.label_data).float()
            print('Transformations ended..')
        else : 
            stacked_np_train = np.expand_dims(np.array(train_data[0]),0)
            
            stacked_np_labels = np.expand_dims(np.array(label_data[0]),0)
            
            for i in range(1,len(train_data)):
                tmp = np.array(train_data[i])
                tmp = np.expand_dims(tmp,0)
                stacked_np_train = np.concatenate((stacked_np_train,tmp))
                tmp = np.array(label_data[i])
                tmp = np.expand_dims(tmp,0)
                stacked_np_labels = np.concatenate((stacked_np_labels,tmp))

            
            self.train_data = torch.from_numpy(stacked_np_train).float()
            self.train_data = torch.permute(self.train_data,(0,3,1,2))
            
            stacked_np_labels = (stacked_np_labels > THRESHOLD_ROAD_BGRD_LABELS) *1.0
            self.label_data = torch.from_numpy(stacked_np_labels).float()
            
            
        self.size = self.train_data.shape[0]
        

    def __getitem__(self, index):
        label = self.label_data[index]
        img = self.train_data[index]
        
        if(self.normalize):
            img = self.normalize(img)
            
        return img,label 

    
    def __len__(self):
        return self.size
    
    def apply_rotation(self,train_data,label_data,augmented_train,augmented_labels):
        '''
            Rotate images in the train dataset by the given angles and add them to the 
            augmented data 
            train_data: training set
            label_data: labels of the training set
            augmented_train: set of the original images plus the transformed ones
            augmented_labels: labels of the set augmented_train
            returns:
                    augmented_train plus the rotated images 
                    labels of the augmented set
            
        '''        
        rotation_degrees = [-90,-60,-45,-15,15,45,60,90]
        for j in range(0,len(train_data)):
            for i in rotation_degrees:
                img = np.array(transformation.rotate(train_data[j],i))
                img = np.expand_dims(img,0)
                augmented_train = np.concatenate((augmented_train,img))
                lb = np.array(transformation.rotate(label_data[j],i))
                lb = np.expand_dims(lb,0)
                augmented_labels = np.concatenate((augmented_labels,lb))
        

        return augmented_train, augmented_labels
    
    def apply_flips(self ,train_data,label_data,augmented_train, augmented_labels): 
        '''
            Apply horizontal and vertical flip to the images in the train dataset and add them to the 
            augmented data 
            train_data: training set
            label_data: labels of the training set
            augmented_train: set of the original images plus the transformed ones
            augmented_labels: labels of the set augmented_train
            returns:
                    augmented_train plus the flipped images 
                    labels of the augmented set
            
        '''        
        for i in range(len(train_data)):
            imgh = np.array(transformation.hflip(train_data[i]))
            imgv = np.array(transformation.vflip(train_data[i]))
            imgh = np.expand_dims(imgh,0)
            imgv = np.expand_dims(imgv,0)
            augmented_train = np.concatenate((augmented_train,imgh))
            augmented_train = np.concatenate((augmented_train,imgv))
            
            lbh = np.array(transformation.hflip(label_data[i]))
            lbv = np.array(transformation.vflip(label_data[i]))
            lbh = np.expand_dims(lbh,0)
            lbv = np.expand_dims(lbv,0)
            augmented_labels = np.concatenate((augmented_labels,lbh))
            augmented_labels = np.concatenate((augmented_labels,lbv))
        

        return augmented_train, augmented_labels
        
    
    def keep_original_images(self,train_data,label_data):
        '''
            Create the augmented set where we concatenate all the transformed images
            train_data: training set
            label_data: labels of the training set
    
            returns:
                    augmented_train set of the original images 
                    labels of the augmented set
            
        '''        
        augmented_train = np.array(train_data[0])
        augmented_train = np.expand_dims(augmented_train,0)
        augmented_labels = np.array(label_data[0])
        augmented_labels = np.expand_dims(augmented_labels,0)
        for i in range(1,len(train_data)):
            tmp = np.array(train_data[i])
            tmp = np.expand_dims(tmp,0)
            tmp2 = np.array(label_data[i])
            tmp2 = np.expand_dims(tmp2,0)
            augmented_train = np.concatenate((augmented_train,tmp))
            augmented_labels = np.concatenate((augmented_labels,tmp2))
            
        return augmented_train,augmented_labels
        
    def apply_transformations(self,train_data,label_data):
        '''
            Apply all the transformations 
            train_data: training set
            label_data: labels of the training set
            augmented_train: set of the original images plus the transformed ones
            augmented_labels: labels of the set augmented_train
            returns:
                    augmented_train plus the transformed images 
                    labels of the set augmented set
            
        '''        
        ## Add original elements 
        augmented_train, augmented_labels = self.keep_original_images(train_data,label_data)
        
        ## Apply rotations
        augmented_train, augmented_labels = self.apply_rotation(train_data,label_data, augmented_train, augmented_labels)
        
        ## Apply H/V Flips
        augmented_train, augmented_labels = self.apply_flips(train_data,label_data,augmented_train,augmented_labels)
        
        return augmented_train, augmented_labels
        

        
        
        
        
        
        
