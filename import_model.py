import model
import torch
import utils as gu
from os import mkdir
from os.path import isdir

# Create an instance of our model
generator = model.generator(5)  # Instantiate our model class

# Load the saved state dictionary into the model
generator.load_state_dict(torch.load('generator.pt'))

# Set the model to evaluation mode

generator.eval()

# sample an image and save it
synth_img = model.sample_fake_img(generator, [1,3,512,512] , n_samples=1)

saving_folder = 'experim/'
if not isdir(saving_folder):
    mkdir(saving_folder)

for i in range(100):
    synth_img = model.sample_fake_img(generator, [1,3,512,512] , n_samples=1)
    gu.SaveImg(saving_folder+'test' + str(i) + '.png', gu.PostProc(synth_img.clone().detach()))