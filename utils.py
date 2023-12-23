import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import vgg as vgg
import wget
import ssl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ssl._create_default_https_context = ssl._create_unverified_context

def ReadImg(imagePath):
    '''
    Read an image as tensor and ensure that it has 3 channel and range from 0 to 255
    output tensImg has dimension [nrow, ncol, nchannel]
    '''
    npImg = plt.imread(imagePath)
    tensImg = torch.tensor(npImg)
    if torch.max(tensImg) <= 1:
        tensImg*=255    
    if len(tensImg.shape) < 3:
        tensImg = tensImg.unsqueeze(2)
        tensImg = torch.cat((tensImg, tensImg, tensImg), 2)
    if tensImg.shape[2] > 3:
        tensImg = tensImg[:,:,:3]
    return tensImg

def ShowImg(tensImg):
    '''
    Show a tensor image
    tensImg dimension should be [nrow, ncol, nchannel]
    '''
    npImg = np.clip((tensImg.data.cpu().numpy())/255, 0,1)
    ax = plt.imshow(npImg)
    return ax

def SaveImg(saveName, tensImg):
    '''
    Show a tensor image as saveName
    tensImg dimension should be [nrow, ncol, nchannel]
    '''
    npImg = np.clip((tensImg.cpu().numpy())/255, 0,1)
    if npImg.shape[2] < 3:
        npImg = npImg[:,:,0]
    plt.imsave(saveName, npImg)
    return 

def PreProc(tensImg):
    '''
    pre-process an image in order to feed it in VGG net
    input: tensImg as dimension [nrow, ncol, nchannel] with channel RGB
    output: normalized preproc image of dimension [1, nchannel, nrow, ncol] with channel BGR
    '''
    out = tensImg[:,:,[2,1,0]] # RGB to BRG
    out = out - torch.tensor([104, 117, 124], device=tensImg.device).view(1,1,3) # substract VGG mean
    return out.permute(2,0,1).unsqueeze(0) # permute and unsqueeze

def PostProc(batchImg):
    '''
    post-process an image in order to display and save it
    input: batchImg as dimension [1, nchannel, nrow, ncol] with channel BGR
    output: post-processed image of dimension [1, nchannel, nrow, ncol] with channel BGR
    '''
    out = batchImg.squeeze(0).permute(1,2,0) # permute and squeeze
    out = out + torch.tensor([104, 117, 124], device=batchImg.device).view(1,1,3) # add VGG mean
    return out[:,:,[2,1,0]] #BRG to RGB    
  
def DownloadVggWeights(modelFolder):
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)
    if not os.path.isfile('./'+modelFolder+'/vgg_conv.pth'):
        defaultPath = os.getcwd()
        os.chdir('./'+modelFolder)
        wget.download('https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth')
        os.chdir(defaultPath)
    return

def CreateVggNet(modelFolder, padding=True):
    DownloadVggWeights(modelFolder)
    vggNet = vgg.CustomVGG(pool='avg', padding=padding)
    vggNet.load_state_dict(torch.load('./'+modelFolder+'/vgg_conv.pth'))
    vggNet.eval()
    for param in vggNet.parameters():
        param.requires_grad = False
    return vggNet



class CustomInceptionV3(nn.Module):
    '''
    InceptionV3 module with custom features extraction
    '''
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(CustomInceptionV3, self).__init__()

        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input)

        # Remove the final classification layers (fc and aux_logits)
        self.inception.fc = nn.Identity()
        if aux_logits:
            self.inception.AuxLogits.fc = nn.Identity()

        self.inception.eval()

        self.outKeys = ['Mixed_7c', 'Mixed_6e', 'Mixed_5d', 'Mixed_4a']

        # Scaling factors for feature normalization
        self.normfeat = [1, 1, 1, 1]

    def forward(self, x):
        out = {}
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        out['Mixed_4a'] = x
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        out['Mixed_5d'] = x
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        out['Mixed_6e'] = x
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        out['Mixed_7c'] = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        out['AvgPool'] = x

        # Ensure all feature maps are squared (e.g., crop or pad)
        for key in out:
            _, c, h, w = out[key].size()
            if h != w:
                # Calculate padding or cropping to make the feature map squared
                diff = abs(h - w)
                pad = diff // 2
                if h > w:
                    out[key] = out[key][:, :, :, pad:pad + w]
                else:
                    out[key] = F.pad(out[key], (pad, pad, 0, 0))

        # Scale features based on self.normfeat
        features = []
        for i, key in enumerate(self.outKeys):
            # Calculate the size for reshaping
            c, size = out[key].size(1), out[key].size(2) * out[key].size(3)
            out[key] = out[key].squeeze(0).reshape(c, -1).transpose(0, 1) / self.normfeat[i]
            features.append(out[key])

        return out