from __future__ import print_function

import os
import pandas as pd
import sys
import argparse, torchvision
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from PIL import Image
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

REL = os.getcwd()
# the following folder is inside crisis computing directory
CHECKPOINT = '/home/shubham/crisis-computing/SupContrast/model_checkpoints/'
VIZ = '/home/shubham/crisis-computing/SupContrast/model_results/'
MODE = 'val' # or 'val'



TRAIN = '../dataset/Train_data/'
TEST = '../dataset/Test_data/'
VAL = '../dataset/Val_data/'

DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (600, 600)

IMG_WIDTH = 600


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='25,45,75',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=128, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    return opt


def set_loader(opt):
    
    # normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=IMG_WIDTH, scale=(0.5, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
        #normalize,
    ])
    
    test_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR),
                                                 transforms.ToTensor()])
                                                     
    train_dataset = datasets.ImageFolder(root=TRAIN,
                                            transform=TwoCropTransform(train_transform))

    val_dataset = datasets.ImageFolder(root=VAL,
                                            transform=TwoCropTransform(test_transform))
    
    test_dataset = datasets.ImageFolder(root=TEST,
                                            transform=test_transform)
                                                     
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)    
                                                     
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
                                                     
    return train_loader, val_loader, test_loader




def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    losses = []
    
    for idx, (images, labels) in enumerate(train_loader):

        images = torch.cat([images[0], images[1]], dim=0)
   
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        bsz = labels.shape[0]

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)
        losses.append(loss.item())
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.average(losses)

def validate(val_loader, model):
    
    features = np.empty(shape=[0, 128])
    val_loss = []
    model.eval()
    targets = []
    with torch.no_grad():
        
        for idx, (images, labels) in enumerate(val_loader):
      
            images = images.to(DEVICE)
            bsz = labels.shape[0]
            targets.extend(labels)

            # compute loss
            feat = model(images)
            features = np.append(features, feat.cpu().numpy(), axis=0)
            
    return features, np.array(targets)

        
def create_features_df(pred_feat, labels):
    
    def create_cols():
        col = 'col'
        cols = []
        for i in range(128):
            cols.append(col + str(i))
        return cols
    
    cols = create_cols()
    feature_df = pd.DataFrame(columns=cols)
    
    for i,rows in enumerate(pred_feat):
        a_series = pd.Series(rows, index=feature_df.columns)
        feature_df = feature_df.append(a_series, ignore_index=True)
    
    targets = pd.DataFrame(labels)
   
    feature_df['target'] = targets

    
    return feature_df
    
def plot_TSNE_2D(feature_df):
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_obj = tsne.fit_transform(feature_df)
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'labels': feature_df['target']})
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x="X", y="Y",
                hue="labels",
                palette=['orange', 'blue'],
                legend='full',
                data=tsne_df)

    plt.savefig(VIZ + 'supcon_embeddings.png')
    plt.show()
    plt.close()
    
    
def model_train_mode(model, optimizer, criterion, train_loader, val_loader, opt):
    
    losses = {'train':[], 'val':[]}
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        train_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        val_loss = validate(val_loader, model, criterion)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print("Train loss value: {:.4f}  Val loss value: {:.4f}".format(train_loss,val_loss))
        print("-------------------------------------")
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)

        if epoch %10 == 0:
            torch.save(model.state_dict(), CHECKPOINT+'supcon_epoch_{}_model.pth'.format(epoch))
        
         
    # save the last model
    torch.save(model.state_dict(), CHECKPOINT+'supcon_final_model.pth')
    print("Training completed!")
    return model, losses
    
def model_test_mode(model, criterion, test_loader):
    return validate(test_loader, model, criterion)
              
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

        self.head = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(2048, 1024),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1024, 128)
            )

    def forward(self, x):
        feat = self.resnet(x)
        feat = torch.nn.functional.normalize(self.head(feat), dim=1)
        return feat

def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model = Resnet().to(DEVICE)
    criterion = SupConLoss().to(DEVICE)
              
    # build optimizer
    optimizer = set_optimizer(opt, model)

    if MODE == "train":
        model, losses = model_train_mode(model, optimizer, criterion, train_loader, val_loader, opt)
    else:
        model.load_state_dict(torch.load(CHECKPOINT + 'supcon_final_model.pth'))
        pred_feat, targets = validate(test_loader, model)
        feature_df = create_features_df(pred_feat, targets)
        
        plot_TSNE_2D(feature_df)
 

if __name__ == '__main__':
    main()
