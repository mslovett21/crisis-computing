# Important libraries
import numpy as np
import torch
import argparse
import sys
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image
import os
# set up GPU
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
MEAN = 0.4905, 0.4729, 0.4560 
STD = 0.2503, 0.2425, 0.2452
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.00001
N_WORKERS = 4
IMAGE_SIZE = (600, 600)
PATH_TRAIN = '/home/shubham/crisis-computing/data/Training_data'
PATH_TEST =  '/home/shubham/crisis-computing/data/Testing_data'
CLASSES = {0:"Informative"  , 1:"Non-Informative"}
CKPT = "/home/shubham/crisis-computing/model_ckpt/"
FINAL_CKPT = '/home/shubham/crisis-computing/model_ckpt/resnet50_bceloss_final_model.pth'

# PATH_TRAIN = '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/data/Training_data'
# PATH_TEST =  '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/data/Testing_data'
# CLASSES = {0:"Informative"  , 1:"Non-Informative"}
# FINAL_CKPT = '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/ResNet50/checkpoints/resnet50_bceloss_final_model.pth'

# Resnet50 Architecture
class Resnet(torch.nn.Module):
  def __init__(self):
    super(Resnet,self).__init__()
    self.resnet = torchvision.models.resnet50(pretrained=True)
    # Following step gives us all layers except last one.
    modules = list(self.resnet.children())[:-1]
    self.resnet = torch.nn.Sequential(*modules)
    # Freeze the model parameters for transfer learning.
    for params in self.resnet.parameters():
      params.requires_grad = False
    
    # Classification head of the model.
    self.head = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(1024, 1, bias=True),
        torch.nn.Sigmoid())
    

  def forward(self, x):
    feat = self.resnet(x)
    output = self.head(feat)
    return feat, output



def transforms_train():
    """
    Returns transformations on training dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """
    transfrms = []
    p = np.random.uniform(0, 1)

    transfrms.append(torchvision.transforms.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR))
    transfrms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    if p >= 0.4 and p <=0.6:
        transfrms.append(torchvision.transforms.ColorJitter(0.2,0.1,0.2))
    elif p < 0.4:
        transfrms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    transfrms.append(torchvision.transforms.ToTensor())  
    transfrms.append(torchvision.transforms.Normalize(MEAN, STD))
    
    return torchvision.transforms.Compose(transfrms)
    

def transforms_test():
    """
    Returns transformations on testing dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(MEAN, STD)])
    return test_transform


def find_mean_std(train_set):
    """
    returns mean and std of the entire dataset
    :params: train_set - train data loader
             test_set - test data loader
    """
    mean = 0.
    var = 0.
    nb_samples = 0.
    
    for data,label in train_set:
        batch_samples = data.size(0) # Batch size
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        nb_samples += batch_samples
   
    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    
    return mean, std

def get_dataloaders(train_transform=None, test_transform=None):
    """
    returns train, validation and test dataloafer objects
    params: train_transform - Augmentation for trainset
            test_transform - Augmentation for testset
            batch_size - size of batch
            n_workers - number of workers
    """
    training = torchvision.datasets.ImageFolder(root=PATH_TRAIN, transform=train_transform)
    testing = torchvision.datasets.ImageFolder(root=PATH_TEST, transform=test_transform)

    train_set = torch.utils.data.DataLoader(training, batch_size = BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    test_set = torch.utils.data.DataLoader(testing, batch_size = BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    return train_set, test_set


def train_loop(model, t_dataset, v_dataset, criterion, optimizer):
    """
    returns loss and accuracy of the model for 1 epoch.
    params: model -  resnet50
          dataset - train or val dataset
          flag - "train" for training, "val" for validation
          criterion - loss function
          optimizer - Adam optimizer
    """
    total = 0
    correct = 0
    epoch_t_loss = 0
    epoch_v_loss = 0
    model.train()
    
    for ind, (image, label) in enumerate(t_dataset):
        image = image.to(DEVICE)
        label = label.type(torch.float).to(DEVICE)

        optimizer.zero_grad()

        _, output = model(image)

        loss = criterion(output, label.unsqueeze(1))
        epoch_t_loss += loss.item()
        predicted = torch.round(output).squeeze(-1) 
        total += label.size(0)
        correct += (predicted==label).sum().item()

        loss.backward()
        optimizer.step()
      
    epoch_t_accuracy = 100*correct/total
    epoch_t_loss = epoch_t_loss/len(t_dataset)
    
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for ind, (image, label) in enumerate(v_dataset):
            image = image.to(DEVICE)
            label = label.type(torch.float).to(DEVICE)


            _, output = model(image)

            loss = criterion(output, label.unsqueeze(1))
            epoch_v_loss += loss.item()
            predicted = torch.round(output).squeeze(-1) 
            total += label.size(0)
            correct += (predicted==label).sum().item()


    epoch_v_accuracy = 100*correct/total
    epoch_v_loss = epoch_v_loss/len(v_dataset)
    
    return epoch_t_loss, epoch_t_accuracy, epoch_v_loss, epoch_v_accuracy




class dataset_loader(Dataset):
  def __init__(self, data, label, transform = None):
    self.data = data
    self.label = label
    self.data_len = len(data)
    self.transform = transform
  
  def __len__(self):
    # returns length of dataset
    return self.data_len
  
  def __getitem__(self, idx):
    # returns image and its label referenced by a specific index.

    if torch.is_tensor(idx):
      idx = idx.tolist()
    image = Image.open(self.data[idx]).convert('RGB')
    # image = np.array(image)
    label = self.label[idx]

    # Apply transformations on the image.
    if self.transform:
      image = self.transform(image)

    return image, label
    

def run_inference(images, labels, flag):
  """
  returns train and test data with augmentation applied.
  """
  model = Resnet().to(DEVICE)
  model.load_state_dict(torch.load(FINAL_CKPT, map_location ='cuda:0'))

  if flag=='test':
    transformation = transforms_test()
    print("Running inference...")
    data = dataset_loader(images, labels, transform=transformation)
    data_test = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
    prob,classes = test(model, data_test)
    print("Inference done!")
    return prob, classes
  else:
    transformation = transforms_train()
    print("Running inference...")
    data = dataset_loader(images, labels, transform=transformation)
    data_train = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
    prob, classes = test(model, data_train)
    print("Inference done!")
    return prob, classes

def train(train, val, model, optimizer, criterion):
    """
    returns train and validation losses of the model over complete training.
    params: train - train dataset
          val - validation dataset
          optimizer - optimizer for training
          criterion - loss function
    """
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    print("Training start...")
    
    for epoch in range(EPOCHS):


        print("Running Epoch {}".format(epoch+1))

        epoch_train_loss, train_accuracy, epoch_val_loss, val_accuracy = train_loop( model, train, val, criterion, optimizer)
        train_losses.append(epoch_train_loss)
        train_acc.append(train_accuracy)
        val_losses.append(epoch_val_loss)
        val_acc.append(val_accuracy)
        
        if (epoch+1)%5==0:
            ckpt_path = CKPT+'resnet50_bceloss_epoch_{}.pth'.format(epoch+1)
            torch.save(model.state_dict(), ckpt_path)

       
  
        print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, train_accuracy))
        print("Val loss: {0:.4f}  Val Accuracy: {1:0.2f}".format(epoch_val_loss, val_accuracy))
        print("--------------------------------------------------------")

    print("Training done...")
    print("Model saved!")
    final_ckpt = CKPT+'resnet50_bceloss_final_model.pth'
    torch.save(model.state_dict(), final_ckpt)

    return train_losses, train_acc, val_losses, val_acc


def test(model, test):
  """
  returns output probabilites and prediction classes
  params: model - model for testing
          test - test dataset
  """
  correct = 0
  total = 0
  predicted_prob = []
  predicted_class = []
  model.eval()
  with torch.no_grad():  
    for image, label in test:

        image = image.to(DEVICE)
        label = label.type(torch.float).to(DEVICE)
        _, output_prob = model(image)
        predicted = torch.round(output_prob).squeeze(-1)
        predicted_prob.extend(output_prob.tolist())
        predicted_class.extend(predicted.tolist())
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    test_accuracy = 100*correct/total
    print('Test Accuracy: %f %%' %(test_accuracy))

  return predicted_prob, predicted_class


def plot_loss(train_losses, val_losses):
    """
    plots train vs validation loss graph
    
    """
    epochs = range(1,EPOCHS+1)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training vs Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Arguments for training/testing')
  parser.add_argument('--flag', type=str, default="test", help=' train or test model')
  opt = parser.parse_args()

  if opt.flag == "train":
    # Initialize model
    model = Resnet().to(DEVICE)

    #Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    #Loss function initialization
    criterion = torch.nn.BCELoss()

  train_transform, test_transform = transforms_train(), transforms_test()
  train_data, test_data = get_dataloaders(train_transform, test_transform)

  if opt.flag == 'train':
    train_loss, test_loss = train(train_data, test_data, model, optimizer, criterion)
    plot_loss(train_loss, test_loss)

  if opt.flag == 'test':
    # load specific model for test
    test_model = Resnet().to(DEVICE)
    test_model.load_state_dict(torch.load(FINAL_CKPT, map_location ='cuda:0'))
    output_prob, output_class = test(test_model, test_data)

