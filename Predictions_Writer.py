import numpy
import re
import matplotlib.image as mpimg
import torch
from PIL import Image
import os 

ROOT_PATH = os.path.abspath(os.curdir)



def requires_dictionary(model_name):
    if model_name == "UNET DEPTH 3" or model_name == "UNET DEPTH 4" or model_name == "UNET DEPTH 5" :
        return False
    else :
        return True  

def mask_to_strings(img_index,image_filename,model,threshold_prediction,device,requires_dictionnary):
    '''Reads a single image and outputs the strings that should go into the submission file'''    
    im = Image.open(image_filename)
    prediction = get_predictions(im,model,threshold_prediction,device,requires_dictionnary)
    im = numpy.array(im,dtype=float)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = prediction[i:i + patch_size, j:j + patch_size]
            labels = patch_to_label(patch,threshold_prediction)
            yield("{:03d}_{}_{},{}".format(img_index, j, i, labels))


def patch_to_label(patch,threshold_prediction):
    '''assign a label to a patch'''    
    df = numpy.mean(patch)
    if df > threshold_prediction:
        return 1
    else:
        return 0

def get_predictions(img,model,threshold_prediction,device,requires_dictionnary):
    '''Evaluate the model to get the predictions for the given image'''    
    model.eval()
    np_array = numpy.array(img,dtype=float)
    data_node = torch.from_numpy(np_array).float()
    data_node = torch.unsqueeze(data_node,dim=0)
    data_node = data_node.permute(0,3,1,2)
    data_node = data_node.to(device)
    if requires_dictionnary:
        output = (torch.sigmoid(model(data_node)['out'])>threshold_prediction)*1.0
    else:
         output = (torch.sigmoid(model(data_node))>threshold_prediction)*1.0
    output = torch.squeeze(output)
    return output.detach().cpu().numpy()


def generate_predictions(model,threshold_prediction,device,model_name):
    '''Converts images into a submission file'''    
    IMG_INDEX = 1
    submission_filename = ROOT_PATH + '/sample_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = ROOT_PATH+'/test_set_images/' + 'test_' + '%d' % i +'/test_' + '%d' % i + '.png'
        print(image_filename)
        image_filenames.append(image_filename)
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:              
            f.writelines('{}\n'.format(s) for s in mask_to_strings(IMG_INDEX,fn,model,threshold_prediction,device,requires_dictionary(model_name)))
            IMG_INDEX += 1
    print('Writing Submission Finished !')