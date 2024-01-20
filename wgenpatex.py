import torch
from torch import nn
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import model
from os import mkdir
from os.path import isdir
import torchvision.transforms as transforms
import utils as gu


#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device("cpu")
print(DEVICE)

def imread(img_name):
    """
    loads an image as torch.tensor on the selected device
    """ 
    np_img = plt.imread(img_name)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=DEVICE)
    if torch.max(tens_img) > 1:
        tens_img/=255
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)
    if tens_img.shape[2] > 3:
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)
    return tens_img.unsqueeze(0)

def imshow(tens_img):
    """
    shows a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
        ax = plt.imshow(np_img)
        ax.set_cmap('gray')
    else:
        ax = plt.imshow(np_img)
    plt.axis('off')
    return plt.show()

def imsave(save_name, tens_img):
    """
    save a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
    plt.imsave(save_name, np_img)
    return 

class gaussian_downsample(nn.Module):
    """
    Downsampling module with Gaussian filtering
    """ 
    def __init__(self, kernel_size, sigma, stride, pad=False):
        super(gaussian_downsample, self).__init__()
        self.gauss = nn.Conv2d(3, 3, kernel_size, stride=stride, groups=3, bias=False)        
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(DEVICE)
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size-1

    def forward(self, x):
        if self.pad:
            x = torch.cat((x, x[:,:,:self.padsize,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize]), 3)
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

class semidual(nn.Module):
    """
    Computes the semi-dual loss between inputy and inputx for the dual variable psi
    """    
    def __init__(self, inputy, device=DEVICE, usekeops=False):
        super(semidual, self).__init__()        
        self.psi = nn.Parameter(torch.zeros(inputy.shape[0], device=device))
        self.yt = inputy.transpose(1,0)
        self.usekeops = usekeops
        self.y2 = torch.sum(self.yt **2,0,keepdim=True)
    def forward(self, inputx):
        if self.usekeops:
            from pykeops.torch import LazyTensor
            y = self.yt.transpose(1,0)
            x_i = LazyTensor(inputx.unsqueeze(1).contiguous())
            y_j = LazyTensor(y.unsqueeze(0).contiguous())
            v_j = LazyTensor(self.psi.unsqueeze(0).unsqueeze(2).contiguous())
            sx2_i = LazyTensor(torch.sum(inputx**2,1,keepdim=True).unsqueeze(2).contiguous())
            sy2_j = LazyTensor(self.y2.unsqueeze(2).contiguous())
            rmv = sx2_i + sy2_j - 2*(x_i*y_j).sum(-1) - v_j
            amin = rmv.argmin(dim=1).view(-1)
            loss = torch.mean(torch.sum((inputx-y[amin,:])**2,1)-self.psi[amin]) + torch.mean(self.psi)
        else:
            cxy = torch.sum(inputx**2,1,keepdim=True) + self.y2 - 2*torch.matmul(inputx,self.yt)
            loss = torch.mean(torch.min(cxy - self.psi.unsqueeze(0),1)[0]) + torch.mean(self.psi)
        return loss
    
class gaussian_layer(nn.Module): 
    """
    Gaussian layer for the dowsampling pyramid
    """ 	   
    def __init__(self, gaussian_kernel_size, gaussian_std, stride = 2, pad=False):
        super(gaussian_layer, self).__init__()
        self.downsample = gaussian_downsample(gaussian_kernel_size, gaussian_std, stride, pad=pad)
    def forward(self, input):
        self.down_img = self.downsample(input)
        return self.down_img

class identity(nn.Module):  
    """
    Identity layer for the dowsampling pyramid
    """   
    def __init__(self):
        super(identity, self).__init__()
    def forward(self, input):
        self.down_img = input
        return input

def create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride = 2, pad=False):
    """
    Create a dowsampling Gaussian pyramid
    """ 
    layer = identity()
    gaussian_pyramid = nn.Sequential(layer)
    for i in range(n_scales-1):
        layer = gaussian_layer(gaussian_kernel_size, gaussian_std, stride, pad=pad)
        gaussian_pyramid.add_module("Gaussian_downsampling_{}".format(i+1), layer)
    return gaussian_pyramid

class patch_extractor(nn.Module):   
    """
    Module for creating custom patch extractor
    """ 
    def __init__(self, patch_size, pad=False):
        super(patch_extractor, self).__init__()
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size-1

    def forward(self, input, batch_size=0):
        if self.pad:
            input = torch.cat((input, input[:,:,:self.padsize,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize]), 3)
        patches = self.im2pat(input).squeeze(0).transpose(1,0)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        return patches

class SemiDualOptimalTransportLayer(nn.Module):
    """
    forward(dualVariablePsi, inputDataX, targetDataY, batchSplitSize=None)
    """
    def __init__(self, targetDataY):
        super(SemiDualOptimalTransportLayer, self).__init__()
        self.targetDataY = targetDataY
        self.numTargetDataY = targetDataY.size(0)
        self.dualVariablePsi = nn.Parameter(torch.zeros(self.numTargetDataY))
        self.squaredY = torch.sum(self.targetDataY.transpose(0,1)**2,0,keepdim=True)

    def forward(self, inputDataX):
        loss = 0
        numInputDataX = inputDataX.size(0)
        dimInputDataX = inputDataX.size(1)
        batchSplitSize = numInputDataX 
        InputDataX = torch.split(inputDataX, batchSplitSize)

        # Compute the J loss function
        for x in InputDataX:            
            costMatrix = (torch.sum(x**2,1,keepdim=True) + self.squaredY - 2*torch.matmul(x,self.targetDataY.transpose(0,1)))/(2)
            loss += torch.sum(torch.min(costMatrix - self.dualVariablePsi.unsqueeze(0),1)[0])
        return loss/numInputDataX + torch.mean(self.dualVariablePsi)


# The following code allows us to Generate a single texture using a combination of Gaussian patches
# and InceptionV3 deep features

def GotexInceptionV32(args):
    
    # get arguments from argparser
    target_img_path = args.target_image_path
    iter_max = args.iter_max
    iter_psi = args.iter_psi
    visu = args.visu
    save = args.save
    img_lr = args.img_lr
    psi_lr = args.psi_lr
    device = args.device

    # parameters for Gaussian downsampling
    gaussian_kernel_size = 4
    gaussian_std = 1
    stride = 2

    # get target image
    target_img = imread(target_img_path)
    
    saving_folder = 'tmp/'

    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        imsave(saving_folder+'original.png', target_img)
    
    # synthesized size
    if args.size is None:
        num_row = target_img.shape[2]
        num_col = target_img.shape[3]
    else:
        num_row = args.size[0]
        num_col = args.size[1]
        
    # visualize every 20 iteration when --visu is True
    monitoring_step = 20
        
    # initialize synthesized image
    target_mean = target_img.view(3,-1).mean(1).to(device) # 1x3x1
    target_std = target_img.view(3,-1).std(1).to(device) # 1x3x1
    synth_img = target_mean.view(1,3,1,1) + 0.05 * target_std.view(1,3,1,1) * torch.randn(1, 3, num_row,num_col, device=device)
    
    synth_img = Variable(synth_img, requires_grad=True)

    # create InceptionV3 feature extractor
    FeatExtractor = gu.CustomInceptionV3().to(device)

    # Ensure it's a PyTorch tensor and is on the appropriate device
    target_img = target_img.to(device)
    
    # extract InceptionV3 features from the target_img
    # The detach() is needed else the InceptionV3 layers will be also trained

    input_features_tmp = FeatExtractor(target_img)

    input_features = []

    for _ , A in input_features_tmp.items():
        input_features.append(A.detach())


    # update normalizing of the InceptionV3 features
    norm_feat = [torch.sqrt(torch.sum(A.detach()**2,1).mean(0)) for A in input_features]
    FeatExtractor.normfeat = norm_feat
    # update input_features
    input_features_tmp = FeatExtractor(target_img)

    input_features = []

    for _ , A in input_features_tmp.items():
        input_features.append(A.detach())
    
    # create psi_optimizers and optimal transport layers
    psi_optimizers = []
    ot_layers  = []
    for i,feat in enumerate(input_features):
        ot_layers.append(SemiDualOptimalTransportLayer(feat.to(device)).to(device))
        psi_optimizers.append(torch.optim.ASGD(ot_layers[i].parameters(), lr=psi_lr, alpha=0.5))

    # create InceptionV3 feature extractor
    FeatExtractor = gu.CustomInceptionV3().to(device)
    norm_feat_device = [n.to(device) for n in norm_feat]
    FeatExtractor.normfeat = norm_feat_device

    n_scales = len(norm_feat)

    # Create Gaussian Pyramid downsamplers
    target_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, 4, stride, pad=False)                  
    input_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, 4, stride, pad=True)
    target_downsampler(target_img) # evaluate on the target image

    # create patch extractors
    target_im2pat = patch_extractor(4, pad=False)
    input_im2pat = patch_extractor(4, pad=True)
    
    
    # create semi-dual module at each scale
    semidual_loss = []
    for s in range(4):
        real_data = target_im2pat(target_downsampler[s].down_img, 2000) # exctract at most n_patches_out patches from the downsampled target images 
        layer = semidual(real_data, device=DEVICE, usekeops=False)
        semidual_loss.append(layer)
        if visu:
            imshow(target_downsampler[s].down_img)

    # Weights on scales
    prop = torch.ones(4, device=DEVICE)/4 # all scales with same weight

    # intialize optimizer for image
    optim_img = torch.optim.Adam([synth_img], lr=0.1)
    
    # initialize the loss vector
    total_loss = np.zeros(iter_max)

    # Main loop
    t = time.time()
    for it in range(iter_max):
    
        # 1. update psi
        
        for itp in range(iter_psi):
            synth_features = [A for _ , A in FeatExtractor(synth_img).items()]
            for i, feat in enumerate(synth_features):
                psi_optimizers[i].zero_grad()
                loss = -ot_layers[i](feat.detach())
                loss.backward()
                # normalize gradient
                ot_layers[i].dualVariablePsi.grad.data /= ot_layers[i].dualVariablePsi.grad.data.norm()
                psi_optimizers[i].step()
        
        
        input_downsampler(synth_img.detach()) # evaluate on the current fake image
        for s in range(4):            
            optim_psi = torch.optim.ASGD([semidual_loss[s].psi], lr=1, alpha=0.5, t0=1)
            for i in range(iter_psi):
                fake_data = input_im2pat(input_downsampler[s].down_img, -1)
                optim_psi.zero_grad()
                loss = -semidual_loss[s](fake_data)
                loss.backward()
                optim_psi.step()
            semidual_loss[s].psi.data = optim_psi.state[semidual_loss[s].psi]['ax']
        
        # 2. perform gradient step on the image
        optim_img.zero_grad()        
        tloss = 0
        
        for s in range(4):
            input_downsampler(synth_img)           
            fake_data = input_im2pat(input_downsampler[s].down_img, -1)
            loss = prop[s]*semidual_loss[s](fake_data)
            loss.backward()
            tloss += loss.item()
        
        for i in range(3):
            synth_features = [A for _ , A in FeatExtractor(synth_img).items()]
            feat = synth_features[i]
            loss = 0.1*ot_layers[i](feat)    
            loss.backward()
            tloss += loss.item()
        
        optim_img.step()

        # save loss
        total_loss[it] = tloss
    
        # save some results
        if it % monitoring_step == 0:
            print('iteration '+str(it)+' - elapsed '+str(int(time.time()-t))+'s - loss = '+str(tloss))
            if visu:
                imshow(synth_img)
            if save:
                imsave(saving_folder+'it'+str(it)+'.png', synth_img)

    print('DONE - total time is '+str(int(time.time()-t))+'s')

    if visu:
        plt.plot(total_loss)
        plt.show()
        if save:
            plt.savefig(saving_folder+'loss_multiscale.png')
        plt.close()
    if save:
        np.save(saving_folder+'loss.npy', total_loss)

    return synth_img


# The following code allows us to Generate a single texture using a combination of Gaussian patches
# and VGG deep features

def GotexVgg(args):
    
    # get arguments from argparser
    target_img_path = args.target_image_path
    iter_max = args.iter_max
    iter_psi = args.iter_psi
    visu = args.visu
    save = args.save
    img_lr = args.img_lr
    psi_lr = args.psi_lr
    device = args.device

    # parameters for Gaussian downsampling
    gaussian_kernel_size = 4
    gaussian_std = 1
    stride = 2

    # get target image
    target_img = gu.ReadImg(target_img_path)
    target_img = gu.PreProc(target_img)
    
    saving_folder = 'tmp/'
    
    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        gu.SaveImg(saving_folder+'original.png', gu.PostProc(target_img))
    
    # synthesized size
    if args.size is None:
        num_row = target_img.shape[2]
        num_col = target_img.shape[3]
    else:
        num_row = args.size[0]
        num_col = args.size[1]
        
    # visualize every 100 iteration when --visu is True
    monitoring_step = 20
        
    # initialize synthesized image
    target_mean = target_img.view(3,-1).mean(1).to(device) # 1x3x1
    target_std = target_img.view(3,-1).std(1).to(device) # 1x3x1
    synth_img = target_mean.view(1,3,1,1) + 0.05 * target_std.view(1,3,1,1) * torch.randn(1, 3, num_row,num_col, device=device)
    
    synth_img = Variable(synth_img, requires_grad=True)
    # synth_img = torch.randn(1,3, num_row, num_col, requires_grad=True, device=device)

    # initialize image optimizer
    image_optimizer = torch.optim.LBFGS([synth_img], lr=img_lr)

    # create VGG feature extractor
    FeatExtractor = gu.CreateVggNet("VggModel", padding=True)
    # extract VGG features from the target_img
    input_features = FeatExtractor(target_img)
    # update normalizing of the VGG features
    norm_feat = [torch.sqrt(torch.sum(A.detach()**2,1).mean(0)) for A in input_features]
    FeatExtractor.normfeat = norm_feat
    # update input_features
    input_features = FeatExtractor(target_img)
    
    # create psi_optimizers and optimal transport layers
    psi_optimizers = []
    ot_layers  = []
    for i,feat in enumerate(input_features):
        ot_layers.append(SemiDualOptimalTransportLayer(feat.to(device)).to(device))
        psi_optimizers.append(torch.optim.ASGD(ot_layers[i].parameters(), lr=psi_lr, alpha=0.5))

    # create VGG feature extractor
    FeatExtractor = gu.CreateVggNet("VggModel", padding=True).to(device)
    norm_feat_device = [n.to(device) for n in norm_feat]
    FeatExtractor.normfeat = norm_feat_device

    n_scales = len(norm_feat)

    # Create Gaussian Pyramid downsamplers
    target_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=False)                  
    input_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=True)
    target_downsampler(target_img) # evaluate on the target image

    # create patch extractors
    target_im2pat = patch_extractor(4, pad=False)
    input_im2pat = patch_extractor(4, pad=True)
    
    

    # create semi-dual module at each scale
    semidual_loss = []
    for s in range(n_scales):
        real_data = target_im2pat(target_downsampler[s].down_img, 2000) # exctract at most n_patches_out patches from the downsampled target images 
        layer = semidual(real_data, device=DEVICE, usekeops=False)
        semidual_loss.append(layer)
        if visu:
            imshow(target_downsampler[s].down_img)

    # Weights on scales
    prop = torch.ones(n_scales, device=DEVICE)/n_scales # all scales with same weight


    
    # initialize loss
    loss_list = [0]*iter_max
    starting_time = time.time()

    # run optimization
    n_iter=[0]
    while n_iter[0] < iter_max:
        
        # The closure is needed since we are using LBFGS optimization
        def closure():       
            # update dual variable psi

            for itp in range(iter_psi):
                synth_features = FeatExtractor(synth_img)
                for i, feat in enumerate(synth_features):
                    psi_optimizers[i].zero_grad()
                    loss = -ot_layers[i](feat.detach())
                    loss.backward()
                    # normalize gradient
                    ot_layers[i].dualVariablePsi.grad.data /= ot_layers[i].dualVariablePsi.grad.data.norm()
                    psi_optimizers[i].step()

            input_downsampler(synth_img.detach()) # evaluate on the current fake image
            for s in range(4):            
                optim_psi = torch.optim.ASGD([semidual_loss[s].psi], lr=1, alpha=0.5, t0=1)
                for i in range(iter_psi):
                    # evaluate on the current fake image
                    fake_data = input_im2pat(input_downsampler[s].down_img, -1)
                    optim_psi.zero_grad()
                    loss = -semidual_loss[s](fake_data)
                    loss.backward()
                    optim_psi.step()
                semidual_loss[s].psi.data = optim_psi.state[semidual_loss[s].psi]['ax']
       
            # update image
            # synth_features = FeatExtractor(synth_img)
            image_optimizer.zero_grad()
            tloss = 0
            
            for i in range(4):
                synth_features = FeatExtractor(synth_img)
                feat = synth_features[i]
                loss = ot_layers[i](feat)    
                loss.backward()
            loss_list[n_iter[0]] = loss.item()
            
            
            input_downsampler(synth_img)
            for s in range(4):        
                fake_data = input_im2pat(input_downsampler[s].down_img, -1)
                loss = 0.00001*prop[s]*semidual_loss[s](fake_data)
                tloss += loss
            
            tloss.backward()

            # monitoring
            if ((n_iter[0]% monitoring_step) == 0):        
                elapsed_time = int(time.time()-starting_time)
                print('iteration = '+str(n_iter[0]))
                print('elapsed time = '+str(elapsed_time)+'s')
                print('OT loss = ' + str(loss.item()))
                if visu:
                    gu.ShowImg(gu.PostProc(synth_img))
                    plt.show()
                if save:
                    gu.SaveImg(saving_folder+'it'+str(n_iter[0])+'.png', gu.PostProc(synth_img.clone().detach()))
            

            n_iter[0]+=1
            return loss

        image_optimizer.step(closure)
      
    return synth_img, loss_list


# The following code allows us to Generate train a CNN as texture generator using a combination of
# Gaussian patches and VGG deep features

def learn_model_VGG(args):

    target_img_name = args.target_image_path
    patch_size = args.patch_size
    n_iter_max = args.n_iter_max
    n_iter_psi = args.n_iter_psi
    n_patches_in = args.n_patches_in
    n_patches_out = args.n_patches_out
    n_scales = args.scales
    usekeops = args.keops
    visu = args.visu
    save = args.save
    
    # fixed parameters
    monitoring_step=50
    saving_folder='tmp/'
    
    # parameters for Gaussian downsampling
    gaussian_kernel_size = 4
    gaussian_std = 1
    stride = 2

    # get target image
    target_img = gu.ReadImg(target_img_name)
    target_img = gu.PreProc(target_img)
    
    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        gu.SaveImg(saving_folder+'original.png', gu.PostProc(target_img))

    # create VGG feature extractor
    FeatExtractor = gu.CreateVggNet("VggModel", padding=True)
    # extract VGG features from the target_img
    input_features = FeatExtractor(target_img)
    # update normalizing of the VGG features
    norm_feat = [torch.sqrt(torch.sum(A.detach()**2,1).mean(0)) for A in input_features]
    FeatExtractor.normfeat = norm_feat
    # update input_features
    input_features = FeatExtractor(target_img)
    
    # create psi_optimizers and optimal transport layers
    psi_optimizers = []
    ot_layers  = []
    for i,feat in enumerate(input_features):
        ot_layers.append(SemiDualOptimalTransportLayer(feat.to(DEVICE)).to(DEVICE))
        psi_optimizers.append(torch.optim.ASGD(ot_layers[i].parameters(), lr=1., alpha=0.5))

    # create VGG feature extractor
    FeatExtractor = gu.CreateVggNet("VggModel", padding=True).to(DEVICE)
    norm_feat_device = [n.to(DEVICE) for n in norm_feat]
    FeatExtractor.normfeat = norm_feat_device

    n_scales = len(norm_feat)

    # Create Gaussian Pyramid downsamplers
    target_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=False)                  
    input_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=True)
    target_downsampler(target_img) # evaluate on the target image

    # create patch extractors
    target_im2pat = patch_extractor(4, pad=False)
    input_im2pat = patch_extractor(4, pad=True)

    # create semi-dual module at each scale
    semidual_loss = []
    for s in range(n_scales):
        real_data = target_im2pat(target_downsampler[s].down_img, 2000) # exctract at most n_patches_out patches from the downsampled target images 
        layer = semidual(real_data, device=DEVICE, usekeops=False)
        semidual_loss.append(layer)
        if visu:
            imshow(target_downsampler[s].down_img)

    # Weights on scales
    prop = torch.ones(n_scales, device=DEVICE)/n_scales # all scales with same weight
    
    # initialize generator
    G = model.generator(n_scales)
    fake_img = model.sample_fake_img(G, target_img.shape, n_samples=1)

    # intialize optimizer for image
    optim_G = torch.optim.Adam(G.parameters(), lr=0.1)
    
    # initialize the loss vector
    total_loss = np.zeros(n_iter_max)

    # Main loop
    t = time.time()
    for it in range(n_iter_max):

        # 1. update psi

        for itp in range(n_iter_psi):
            synth_features = FeatExtractor(fake_img)
            for i, feat in enumerate(synth_features):
                psi_optimizers[i].zero_grad()
                loss = -ot_layers[i](feat.detach())
                loss.backward()
                # normalize gradient
                ot_layers[i].dualVariablePsi.grad.data /= ot_layers[i].dualVariablePsi.grad.data.norm()
                psi_optimizers[i].step()


        fake_img = model.sample_fake_img(G, target_img.shape, n_samples=1)
        input_downsampler(fake_img.detach())
        
        for s in range(4):            
            optim_psi = torch.optim.ASGD([semidual_loss[s].psi], lr=1, alpha=0.5, t0=1)

            for i in range(n_iter_psi):
                 # evaluate on the current fake image
                fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in)
                optim_psi.zero_grad()
                loss = -semidual_loss[s](fake_data)
                loss.backward()
                optim_psi.step()
            semidual_loss[s].psi.data = optim_psi.state[semidual_loss[s].psi]['ax']

        # 2. perform gradient step on the image
        optim_G.zero_grad()
        tloss = 0


        input_downsampler(fake_img)
        
        for s in range(5):
            synth_features = FeatExtractor(fake_img)
            feat = synth_features[s]
            loss = ot_layers[s](feat)
            tloss += loss

        tloss.backward(retain_graph=True)
        
        for s in range(4):        
            fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in)
            loss = 0.00001*prop[s]*semidual_loss[s](fake_data)
            tloss += loss
        
        tloss.backward()
        
        optim_G.step()

        # save loss
        total_loss[it] = tloss.item()
    
        # monitoring
        if ((it% monitoring_step) == 0):        
            elapsed_time = int(time.time()-t)
            print('iteration = '+str(it))
            print('elapsed time = '+str(elapsed_time)+'s')
            print('OT loss = ' + str(loss.item()))
            if visu:
                gu.ShowImg(gu.PostProc(fake_img))
                plt.show()
            if save:
                gu.SaveImg(saving_folder+'it'+str(it)+'.png', gu.PostProc(fake_img.clone().detach()))

    print('DONE - total time is '+str(int(time.time()-t))+'s')

    if visu:
        gu.ShowImg(gu.PostProc(fake_img))
        plt.show()
        plt.pause(0.01)
        if save:
            gu.SaveImg(saving_folder+'it'+str(it)+'.png', gu.PostProc(fake_img.clone().detach()))
        plt.close()
    if save:
        np.save(saving_folder+'loss.npy', total_loss)
        
    return G


# The following code allows us to Generate train a CNN as texture generator using a combination of
# Gaussian patches and InceptionV3 deep features

def learn_model_incep(args):

    target_img_name = args.target_image_path
    patch_size = args.patch_size
    n_iter_max = args.n_iter_max
    n_iter_psi = args.n_iter_psi
    n_patches_in = args.n_patches_in
    n_patches_out = args.n_patches_out
    n_scales = args.scales
    usekeops = args.keops
    visu = args.visu
    save = args.save
    
    # fixed parameters
    monitoring_step=50
    saving_folder='tmp/'
    
    # parameters for Gaussian downsampling
    gaussian_kernel_size = 4
    gaussian_std = 1
    stride = 2

    # get target image
    target_img = imread(target_img_name)
    
    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        imsave(saving_folder+'original.png', target_img)

    # create InceptionV3 feature extractor
    FeatExtractor = gu.CustomInceptionV3().to(DEVICE)

    # Ensure it's a PyTorch tensor and is on the appropriate device
    target_img = target_img.to(DEVICE)
    
    # extract InceptionV3 features from the target_img

    input_features_tmp = FeatExtractor(target_img)

    input_features = []

    # The detach() is needed else the InceptionV3 layers will be also trained

    for _ , A in input_features_tmp.items():
        input_features.append(A.detach())


    # update normalizing of the VGG features
    norm_feat = [torch.sqrt(torch.sum(A.detach()**2,1).mean(0)) for A in input_features]
    FeatExtractor.normfeat = norm_feat
    # update input_features
    input_features_tmp = FeatExtractor(target_img)

    input_features = []

    for _ , A in input_features_tmp.items():
        input_features.append(A.detach())
    
    # create psi_optimizers and optimal transport layers
    psi_optimizers = []
    ot_layers  = []
    for i,feat in enumerate(input_features):
        ot_layers.append(SemiDualOptimalTransportLayer(feat.to(DEVICE)).to(DEVICE))
        psi_optimizers.append(torch.optim.ASGD(ot_layers[i].parameters(), lr=1., alpha=0.5))

    # create VGG feature extractor
    FeatExtractor = gu.CustomInceptionV3().to(DEVICE)
    norm_feat_device = [n.to(DEVICE) for n in norm_feat]
    FeatExtractor.normfeat = norm_feat_device

    n_scales = len(norm_feat)

    # Create Gaussian Pyramid downsamplers
    target_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=False)                  
    input_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride, pad=True)
    target_downsampler(target_img) # evaluate on the target image

    # create patch extractors
    target_im2pat = patch_extractor(4, pad=False)
    input_im2pat = patch_extractor(4, pad=True)

    # create semi-dual module at each scale
    semidual_loss = []
    for s in range(n_scales):
        real_data = target_im2pat(target_downsampler[s].down_img, 2000) # exctract at most n_patches_out patches from the downsampled target images 
        layer = semidual(real_data, device=DEVICE, usekeops=False)
        semidual_loss.append(layer)
        if visu:
            imshow(target_downsampler[s].down_img)

    # Weights on scales
    prop = torch.ones(n_scales, device=DEVICE)/n_scales # all scales with same weight
    
    # initialize generator
    G = model.generator(n_scales)
    fake_img = model.sample_fake_img(G, target_img.shape, n_samples=1)

    # intialize optimizer for image
    optim_G = torch.optim.Adam(G.parameters(), lr=0.1)
    
    # initialize the loss vector
    total_loss = np.zeros(n_iter_max)

    # Main loop
    t = time.time()
    for it in range(n_iter_max):

        # 1. update psi
        fake_img = model.sample_fake_img(G, target_img.shape, n_samples=1)

        for itp in range(n_iter_psi):
            synth_features = [A for _ , A in FeatExtractor(fake_img).items()]
            for i, feat in enumerate(synth_features):
                psi_optimizers[i].zero_grad()
                loss = -ot_layers[i](feat.detach())
                loss.backward()
                # normalize gradient
                ot_layers[i].dualVariablePsi.grad.data /= ot_layers[i].dualVariablePsi.grad.data.norm()
                psi_optimizers[i].step()


        input_downsampler(fake_img.detach())
        
        for s in range(4):            
            optim_psi = torch.optim.ASGD([semidual_loss[s].psi], lr=1, alpha=0.5, t0=1)

            for i in range(n_iter_psi):
                 # evaluate on the current fake image
                fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in)
                optim_psi.zero_grad()
                loss = -semidual_loss[s](fake_data)
                loss.backward()
                optim_psi.step()
            semidual_loss[s].psi.data = optim_psi.state[semidual_loss[s].psi]['ax']

        # 2. perform gradient step on the image
        optim_G.zero_grad()
        tloss = 0

        for s in range(4):
            input_downsampler(fake_img)           
            fake_data = input_im2pat(input_downsampler[s].down_img, -1)
            loss = prop[s]*semidual_loss[s](fake_data)
            tloss += loss

        tloss.backward(retain_graph=True)
        
        for i in range(3):
            synth_features = [A for _ , A in FeatExtractor(fake_img).items()]
            feat = synth_features[i]
            loss = 0.1*ot_layers[i](feat)  
            tloss += loss
        
        tloss.backward()

        optim_G.step()

        # save loss
        total_loss[it] = tloss.item()
    
        # save some results
        if it % monitoring_step == 0:
            print('iteration '+str(it)+' - elapsed '+str(int(time.time()-t))+'s - loss = '+str(tloss.item()))
            if visu:
                imshow(fake_img)
            if save:
                imsave(saving_folder+'it'+str(it)+'.png', fake_img)

    print('DONE - total time is '+str(int(time.time()-t))+'s')

    if visu:
        plt.plot(total_loss)
        plt.show()
        if save:
            plt.savefig(saving_folder+'loss_multiscale.png')
        plt.close()
    if save:
        np.save(saving_folder+'loss.npy', total_loss)
        
    return G
