import os
import numpy as np
import Model
import random
import numpy as np
import Loading_helpers

''' This module is used to tune the hyperparameters of the model
'''
num_samples = 20

def find_best_params(model,train_loader,validation_loader,device,search = 'Random Search'):
    '''The goal of this function is to find the best hyperparameters for our model
    model: the model for which we want to search for the best hyperparameters.
    search: Parameter to choose between Grid search or random search
    returns: 
        The hyperparameters that give the best F1-score.
        A dictionary that contains all the tested hyperparameters with its corresponding F1-score 
        to trace the results of every combination of hyperparameters.
    '''
    lr_values = [0.00025, 0.0001, 0.0002,0.0003,0.001,0.1]
    nb_epochs_values = [20,30,50,60,75,100,120]
    sigmoid_threshold_values = [ 0.35,0.40,0.45,0.5 ]
    
    lr_tune = []
    nb_epochs_tune = []
    thresholds = []
    F1_scores = []
    
    if(search == 'Grid Search'):
    
        for lr in lr_values:
            for nb_epochs in nb_epochs_values:
                for sgt in sigmoid_threshold_values:
                    _,acc = Model.train_model(model,nb_epochs,train_loader,validation_loader,device,lr,sgt,dictionnary=True)
                    print(f'Testing for learning rate: {lr} and number of epochs: {nb_epochs} and sigmoid threshold: {sgt}')
                    lr_tune.append(lr)
                    nb_epochs_tune.append(nb_epochs)
                    thresholds.append(sgt)
                    F1_scores.append(acc)
    elif search == 'Random Search': 
        
        for i in range(num_samples):
            lr = random.choice(lr_values)
            nb_epochs = random.choice(nb_epochs_values)
            sgt = random.choice(sigmoid_threshold_values)
            _,acc = Model.train_model(model,nb_epochs,train_loader,validation_loader,device,lr,sgt,dictionnary=True)
            print(f'Testing for learning rate: {lr} and number of epochs: {nb_epochs} and sigmoid threshold: {sgt}')
            lr_tune.append(lr)
            nb_epochs_tune.append(nb_epochs)
            thresholds.append(sgt)
            F1_scores.append(acc)

    else:
        print("The available tunning algorithms are Grid Search and Random Search")
        
    tune_analyse = {
        'learning rate': lr_tune,
        'nb of epochs': nb_epochs_tune,
        'thresholds': thresholds,
        'F1 scores': F1_scores
    }
    
    id_best= np.argmax(tune_analyse['F1 scores'])
    best_hyp = [tune_analyse['learning rate'][id_best],tune_analyse['nb of epochs'][id_best],tune_analyse['thresholds'][id_best] ]
    print('Best hyperparameters : ' , best_hyp)
    
    print('Tuning complete')
    return tune_analyse , best_hyp
    
