import argparse
from all_functions import learn_model_VGG
import all_functions
import model
import torch
import utils as gu
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('target_image_path', help='paths of target texture image')
parser.add_argument('-w', '--patch_size', type=int,default=4, help="patch size (default: 4)")
parser.add_argument('-nmax', '--n_iter_max', type=int, default=5000, help="max iterations of the algorithm(default: 5000)")
parser.add_argument('-npsi', '--n_iter_psi', type=int, default=20, help="max iterations for psi (default: 20)")
parser.add_argument('-plr', '--psi_lr', type=float, default=1., help="learning rate for the SGD psi optimizer (default: 1.)")
parser.add_argument('-ilr', '--img_lr', type=float, default=0.1, help="learning rate for the ADAM image optimizer (default: 0.1)")
parser.add_argument('-nin', '--n_patches_in', type=int, default=-1, help="number of patches of the synthetized texture used at each iteration, -1 corresponds to all patches (default: -1)")
parser.add_argument('-nout', '--n_patches_out', type=int, default=2000, help="number maximum of patches of the target texture used, -1 corresponds to all patches (default: 2000)")
parser.add_argument('-sc', '--scales', type=int, default=4, help="number of scales used (default: 4)")
parser.add_argument('-ly', '--layers', type=int, default=5, help="number of layers used (default: 5)")
parser.add_argument('-gw', '--Gauss_weight', type=float, default=0.00001, help="weight given by loss function to Gaussian patches (default: 0.00001)")
parser.add_argument('--visu',  action='store_true', help='show intermediate results')
parser.add_argument('--save',  action='store_true', help='save temp results in /tmp folder')
parser.add_argument('--keops', action='store_true', help='use keops package')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('selected device is '+str(device))

generator = learn_model_VGG(args)

# save the texture generator
torch.save(generator.state_dict(), 'generator.pt')

# sample an image and save it
synth_img = model.sample_fake_img(generator, [1,3,512,512] , n_samples=1)
gu.SaveImg('tmp/'+'it-last'+'.png', gu.PostProc(synth_img.clone().detach()))
ax = plt.imshow(np.clip(gu.PostProc(synth_img.cpu().clone().detach())/255, 0,1))
ax.set_cmap('gray')
plt.axis('off')
plt.show()