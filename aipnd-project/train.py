"Right"
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import datasets , transforms , models
import argparse


import signal

from contextlib import contextmanager

import requests


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable
        
        
def main():
    
    
    args = get_input_args()
    dataloader , image_set = datasets(args.data_dir)
    model,crit , opti = train_model_set(args.acrh,args.hidden_unit,args.learning_rate)
    #test_loss , accuracy = validation(model,test_loss , crit)
    train_validation_save(model,image_set,dataloader,crit,opti,args.epochs,args.gpu,args.save_dir)
    
   




def get_input_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type = str ,  help= 'images directory')
    parser.add_argument('--save_dir' , type = str , default = 'checkpoint' , help="Save checkpoint in file")
    parser.add_argument('--acrh' , type = str , default = 'densenet121' , help = ['vgg13 / densenet121 / densenet161'])
    
    parser.add_argument('--learning_rate' , type = float , default = 0.0001 , help = 'Learning Rate')
    parser.add_argument('--hidden_unit' , type = int , default = [768,384] ,nargs = '+' , help = 'hidden_unit[xx,xx]')
    parser.add_argument('--epochs' , type = int , default = 10 , help = 'Epochs')
    parser.add_argument('--gpu' , type = str  , default = 'cpu' , help = 'Use GPU / CPU ')
    
    return parser.parse_args()



def datasets(data_dir):
    import torch
    from torchvision import datasets , transforms , models
    
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485,0.456,0.406),
                                                               (0.229,0.224,0.225))])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(data_dir,transform = data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets,batch_size = 40,shuffle = True)
    return dataloaders,image_datasets

def train_model_set(arc,hidden_layers,learn_rate):
    from collections import OrderedDict
    if arc == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif arc == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif arc == 'densenet161':
        model = models.densenet161(pretrained = True)
        

    for param in model.parameters():
        param.requires_grad=False
    
    
    if len(hidden_layers) > 1:
        h1 , h2 = hidden_layers[0],hidden_layers[1]
        classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(1024,h1)),
                                                ('relu1',nn.ReLU()),
                                                ('fc2',nn.Linear(h1,h2)),
                                                ('relu2',nn.ReLU()),
                                                ('fc3',nn.Linear(h2,102)),
                                                ('output',nn.LogSoftmax(dim = 1))]))
    else :    
        h1 = hidden_layers
        classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(1024,h1)),
                                                ('relu1',nn.ReLU()),
                                                ('fc2',nn.Linear(h1,102)),
                                                ('output',nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learn_rate)
    
    return model,criterion , optimizer

def validation(model,testloader,criterion,train_tool):    
    test_loss = 0
    accuracy = 0
    for ii , (images,labels) in enumerate(testloader):
        images , labels = images.to(train_tool) , labels.to(train_tool)
        
        output = model.forward(images)
        test_loss += criterion(output,labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim = 1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss , accuracy

def train_validation_save(model,image_set,dataloader,criterion,optimizer,epochs,train_tool,save_checkpoint_file):        
    import torch
    from torchvision import datasets , transforms , models
    test_transforms = transforms.Compose([transforms.RandomRotation(20),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485,0.456,0.406),
                                                          (0.229,0.224,0.225))])

    test_set = datasets.ImageFolder('/home/workspace/aipnd-project/flowers/valid' , transform = test_transforms)

    testloader = torch.utils.data.DataLoader(test_set,batch_size = 40,shuffle=False)
    
    
    epochs = epochs
    every_print = 40
    steps = 0
    if train_tool == 'gpu':
        train_tool = 'cuda'
    elif train_tool =='cuda':
        train_tool == 'cuda'
    else :
        train_tool = 'cpu'
    model.to(train_tool)
    
    for e in keep_awake(range(epochs)):
        running_loss = 0
        for images , labels in dataloader:
            images , labels = images.to(train_tool) , labels.to(train_tool)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % every_print == 0:
                with torch.no_grad():
                    test_loss , accuracy = validation(model, testloader, criterion,train_tool)
                print('Epochs : %s / %s ' % (e+1,epochs),
                      'Running_loss : %.4f' % (running_loss / every_print),
                      'Test Loss : %.4f' % (test_loss/len(testloader)),
                      'Accuracy : %.4f' % (accuracy/len(testloader)))
                      
                running_loss = 0
    
    
    

    
    model.class_to_idx = image_set.class_to_idx
    model.to('cpu')
    torch.save(model.state_dict(), save_checkpoint_file+'.pth')
    
if __name__ == "__main__":
    main()
                    