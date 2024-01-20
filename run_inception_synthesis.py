import argparse
import wgenpatex
import torch

parser = argparse.ArgumentParser()
parser.add_argument('target_image_path', help='paths of target texture image')
parser.add_argument('-s', '--size', default=None, help="size of synthetized texture [nrow, ncol] (default: target texture size)")
parser.add_argument('-nmax', '--iter_max', type=int, default=1000, help="max iterations of the algorithm(default: 1000)")
parser.add_argument('-npsi', '--iter_psi', type=int, default=10, help="max iterations for psi (default: 10)")
parser.add_argument('-ilr', '--img_lr', type=float, default=1., help="learning rate for the LBFGS image optimizer (default: 1.)")
parser.add_argument('-plr', '--psi_lr', type=float, default=1., help="learning rate for the SGD psi optimizer (default: 1.)")
parser.add_argument('-nout', '--n_patches_out', type=int, default=2000, help="number maximum of patches of the target texture used, -1 corresponds to all patches (default: 2000)")
parser.add_argument('--visu',  action='store_true', help='show intermediate results')
parser.add_argument('--save',  action='store_true', help='save intermediate results in /tmp folder')
args = parser.parse_args()

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('selected device is '+str(device))
args.device = device

synth_img = wgenpatex.GotexInceptionV3(args)


# plot and save the synthesized texture 
wgenpatex.imshow(synth_img)
wgenpatex.imsave('synthesized.png', synth_img)