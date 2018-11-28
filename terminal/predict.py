import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sns
from torch.autograd import Variable


def main():
    
    args = get_args()
    model = load_checkpoint(args.checkpoint)
    model.to(args.gpu)
    cat_to_name = load_category_list(args.category_names)
    plot_solution(cat_to_name,args.input_image, model,args.top_k)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_image',type = str , help = 'image path')
    parser.add_argument('checkpoint',type = str , help = 'checkpoint path')
    parser.add_argument('--top_k',type = int , default = 5 , help = 'the number of top_k')
    parser.add_argument('--category_names' , type = str , default = '/home/workspace/aipnd-project/cat_to_name.json' 
                        ,help = 'category list path ')
    parser.add_argument('--gpu', type = str , default = 'cuda',help = 'Use GPU or CPU ')
    
    return parser.parse_args()

def load_checkpoint(checkpoint):
    check_p =torch.load(checkpoint)
    hidden_size = check_p['hidden_size']
    if check_p['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if len(hidden_size) > 1:
            h1 , h2 = hidden_size[0],hidden_size[1]
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,h1)),
                                                ('relu1',nn.ReLU()),
                                                ('fc2',nn.Linear(h1,h2)),
                                                ('relu2',nn.ReLU()),
                                                ('fc3',nn.Linear(h2,102)),
                                                ('output',nn.LogSoftmax(dim = 1))]))
        else :    
            h1 = hidden_size
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,h1)),
                                                ('relu1',nn.ReLU()),
                                                ('fc2',nn.Linear(h1,102)),
                                                ('output',nn.LogSoftmax(dim = 1))]))    
            
    elif check_p['arch'] == 'densenet121':
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False  
        if len(hidden_size) > 1:
            h1 , h2 = hidden_size[0],hidden_size[1]
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(1024,h1)),
                                                ('relu1',nn.ReLU()),
                                                ('fc2',nn.Linear(h1,h2)),
                                                ('relu2',nn.ReLU()),
                                                ('fc3',nn.Linear(h2,102)),
                                                ('output',nn.LogSoftmax(dim = 1))]))
        else :    
            h1 = hidden_size
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(1024,h1)),
                                                ('relu1',nn.ReLU()),
                                                ('fc2',nn.Linear(h1,102)),
                                                ('output',nn.LogSoftmax(dim = 1))]))
             
    else :
        print("it's a undefined architecture")
        
        
    
    model.classifier = classifier
    model.load_state_dict(check_p['state_dict'])
    model.class_to_idx = check_p['class_to_idx']
    return model
    
    

def load_category_list(category_names):
    import json
    with open(category_names,'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    
    image_path = image
    img = Image.open(image)
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
  
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(name_list,image_path, model, topk):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)
    output = model.forward(model_input)
    probs = torch.exp(output)
    #model.to('cuda')
    top_probs, top_labs = probs.topk(topk)
    #top_probs , top_labs = top_probs.to('cpu') , top_labs.to('cpu')
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [name_list[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels,top_flowers


def plot_solution(name_list,image_path, model,topk):
 
    image_path = str(image_path)
    flower_num = (image_path.split('/'))[-2]
 
    img = process_image(image_path)

    probs, labs,flowers= predict(name_list,image_path, model,topk)
    for prob, flower_name in zip(probs, flowers):
        print('%20s : %.4f' % (flower_name, prob))


if __name__ == "__main__":
    main()
