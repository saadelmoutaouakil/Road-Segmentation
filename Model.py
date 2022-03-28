import time
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os 

'''
This module train the given model and evaluate it on the validation set. 
'''

ROOT_PATH = os.path.abspath(os.curdir)

#Some models ouputs OrderedDict, while others don't. This helper function sets the correct flag.
def training_requires_dictionary(model_name):
    if model_name == "UNET DEPTH 3" or model_name == "UNET DEPTH 4" or model_name == "UNET DEPTH 5" :
        return False
    else :
        return True    

def f1_score(tp,pos_pred,pos_labels):
    if (pos_pred == 0) or (pos_labels == 0):
        return 0

    precision = tp / pos_pred
    recall = tp / pos_labels
    return (2 * precision * recall) / (precision + recall)


## Trains the model and returns the best model and its F1-Score. 
def train_model(model,nb_epochs,train_load,validation_load,device,lr,sigmoid_threshold,model_name,saving_path,optimized_for_AICrowd = False,save_model = False):

    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    for ep in range(nb_epochs):
        print('Running Epoch ', ep)

        for i,(inputs, labels) in enumerate(train_load):
            inputs,labels= inputs.float(),labels.float()
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if(training_requires_dictionary(model_name)):
                outputs = torch.squeeze(outputs['out'])
            else:
                outputs = torch.squeeze(outputs)
            preds = (torch.sigmoid(outputs)>sigmoid_threshold) * 1
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        ## Validation Step
        if not optimized_for_AICrowd :
            current_acc = 0
            positive_prediction = 0
            positive_labels = 0
            tp = 0

            for i,(inputs, labels) in enumerate(validation_load):
                model.eval()   
                inputs,labels= inputs.float(),labels.float()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if(training_requires_dictionary(model_name)):
                    outputs = torch.squeeze(outputs['out'])
                else:
                    outputs = torch.squeeze(outputs)
                preds = (torch.sigmoid(outputs)>sigmoid_threshold)*1
            
                positive_prediction += torch.sum(preds == 1).item()
                positive_labels += torch.sum(labels == 1).item()
                p = (preds == labels)
                tp += torch.sum( p & (labels == 1)).item()
        

            current_acc = f1_score(tp,positive_prediction,positive_labels)
            print('f1_score : ', current_acc)
            if current_acc > best_acc:
                best_acc = current_acc
                best_model = copy.deepcopy(model.state_dict())
                
    if optimized_for_AICrowd :
        best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    if save_model :
        torch.save(model.state_dict(),saving_path)
    return model,best_acc

