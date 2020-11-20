# Important libraries
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image
import os
# set up GPU
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
MEAN = 0.5,0.5,0.5
STD = 0.5,0.5,0.5
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.00001
N_WORKERS = 12
IMAGE_SIZE = (600, 600)
PATH_TRAIN = '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/data/Training_data'
PATH_TEST =  '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/data/Testing_data'
CLASSES = {0:"Informative"  , 1:"Non-Informative"}
FINAL_CKPT = '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/ResNet50/checkpoints/resnet50_bceloss_final_model.pth'

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
        torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.BatchNorm1d(2048),
        torch.nn.Linear(2048, 1, bias=True),
        torch.nn.Sigmoid())
    

  def forward(self, x):
    x = self.resnet(x)
    x = self.head(x)
    return x



def transforms_train():
    """
    Returns transformations on training dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.6),
                      torchvision.transforms.RandomAffine(degrees=10,shear=(0.05,0.15)),
                      torchvision.transforms.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(MEAN, STD)])
    
    return train_transform

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


def get_dataloaders(train_transform, test_transform):
    """
    returns train, validation and test dataloafer objects
    params: train_transform - Augmentation for trainset
            test_transform - Augmentation for testset
            batch_size - size of batch
            n_workers - number of workers
    """
    training = torchvision.datasets.ImageFolder(root=PATH_TRAIN, transform=train_transform)
    validation = torchvision.datasets.ImageFolder(root=PATH_TRAIN, transform=test_transform)
    testing = torchvision.datasets.ImageFolder(root=PATH_TEST, transform=test_transform)

    # sample part data for validation pupose. Train-val split = 80-20
    n_train = len(training)
    indices = list(range(n_train))
    split = int(np.floor(0.2*n_train))
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)


    train_set = torch.utils.data.DataLoader(training,batch_size = BATCH_SIZE,  sampler=train_sampler, num_workers=N_WORKERS)
    val_set = torch.utils.data.DataLoader(training, batch_size = BATCH_SIZE, sampler=valid_sampler, num_workers=N_WORKERS)
    test_set = torch.utils.data.DataLoader(testing, batch_size = BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    return train_set, val_set, test_set

def train_loop(model, dataset, flag):
  """
  returns loss and accuracy of the model for 1 epoch.
  params: model -  resnet50
          dataset - train or val dataset
          flag - "train" for training, "val" for validation
  """
  total = 0
  correct = 0
  epoch_loss = 0
  for ind, (image, label) in enumerate(dataset):
      image = image.to(DEVICE)
      label = label.type(torch.float).to(DEVICE)

      if flag == "train":
        optimizer.zero_grad()
      
      output = model(image)
      
      loss = criterion(output, label.unsqueeze(1))
      epoch_loss += loss.item()
      predicted = torch.round(output).squeeze(-1) 
      total += label.size(0)
      correct += (predicted==label).sum().item()
      

      if flag=="train":
        loss.backward()
        optimizer.step()

  epoch_accuracy = 100*correct/total
  epoch_loss = epoch_loss/len(dataset)
  
  return epoch_loss, epoch_accuracy


def train(train, val, model, optimizer, criterion):
  """
  returns train and validation losses of the model over complete training.
  params: train - train dataset
          val - validation dataset
          optimizer - optimizer for training
          criterion - loss function
  """
  train_losses = []
  val_losses = []
  print("Training start...")

  for epoch in range(EPOCHS):

    model = model.train()

    print("Running Epoch {}".format(epoch+1))
    
    epoch_train_loss, train_accuracy = train_loop(model,train, "train")
    train_losses.append(epoch_train_loss)
  
    if (epoch+1)%25==0:
       ckpt_path = '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/ResNet50/checkpoints/resnet50_bceloss_epoch_{}.pth'.format(epoch+1)
       torch.save(model.state_dict(), ckpt_path)

    model = model.eval()
    with torch.no_grad():
      epoch_val_loss, validation_accuracy = train_loop(model, val, "val")
      val_losses.append(epoch_val_loss)
  
    print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, train_accuracy))
    print("Validation loss: {0:.4f}  Validation Accuracy: {1:0.2f}".format(epoch_val_loss, validation_accuracy))
    print("--------------------------------------------------------")

  print("Training done...")
  print("Model saved!")
  final_ckpt = '/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/ResNet50/checkpoints/resnet50_bceloss_final_model.pth'
  torch.save(model.state_dict(), final_ckpt)

  return train_losses, val_losses

def test(model, test):
  """
  returns output probabilites and prediction classes
  params: model - model for testing
          test - test dataset
  """
  correct = 0
  total = 0
  model.eval()
  with torch.no_grad():  
    for image, label in test:

        image = image.to(DEVICE)
        label = label.type(torch.float).to(DEVICE)
        output_prob = model(image)
        predicted = torch.round(output_prob).squeeze(-1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    test_accuracy = 100*correct/total
    print('Test Accuracy: %f %%' %(test_accuracy))

  return output_prob, predicted


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
  
  if DEVICE=="cuda":
    print("GPU found!")
  else:
    print("No GPU found...")
  # Initialize model
  model = Resnet().to(DEVICE)
  
  #Optimizer initialization
  optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

  #Loss function initialization
  criterion = torch.nn.BCELoss()

  train_transform, test_transform = transforms_train(), transforms_test()
  train_data, val_data, test_data = get_dataloaders(train_transform, test_transform)

  train_loss, val_loss = train(train_data, val_data, model, optimizer, criterion)
  plot_loss(train_loss, val_loss)

  # load specific model for test
  test_model = Resnet().to(DEVICE)
  test_model = test_model.load_state_dict(torch.load(FINAL_CKPT))
  output_prob, output_class = test(test_model, test_data)

