import model
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir

def imsave(save_name, tens_img):
    """
    save a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
    plt.imsave(save_name, np_img)
    return 

# Create an instance of our model
generator = model.generator(5)

# Load the saved state dictionary into the model
generator.load_state_dict(torch.load('generator.pt'))

# Set the model to evaluation mode
generator.eval()

saving_folder = 'experim/'
if not isdir(saving_folder):
    mkdir(saving_folder)

for i in range(100):
    synth_img = model.sample_fake_img(generator, [1,3,1024,1024] , n_samples=1)
    imsave(saving_folder+'test_5_' + str(i) + '.png',synth_img.clone().detach())