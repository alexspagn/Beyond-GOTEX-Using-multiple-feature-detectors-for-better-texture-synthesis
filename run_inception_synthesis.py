import argparse
from all_functions import GotexInceptionV3
import all_functions
import torch

parser = argparse.ArgumentParser()
parser.add_argument('target_image_path', help='paths of target texture image')
parser.add_argument('-s', '--size', default=None, help="size of synthetized texture [nrow, ncol] (default: target texture size)")
parser.add_argument('-w', '--patch_size', type=int,default=4, help="patch size (default: 4)")
parser.add_argument('-nmax', '--iter_max', type=int, default=1000, help="max iterations of the algorithm(default: 1000)")
parser.add_argument('-npsi', '--iter_psi', type=int, default=20, help="max iterations for psi (default: 20)")
parser.add_argument('-ilr', '--img_lr', type=float, default=0.1, help="learning rate for the ADAM image optimizer (default: 0.1)")
parser.add_argument('-plr', '--psi_lr', type=float, default=1., help="learning rate for the SGD psi optimizer (default: 1.)")
parser.add_argument('-nout', '--n_patches_out', type=int, default=2000, help="number maximum of patches of the target texture used, -1 corresponds to all patches (default: 2000)")
parser.add_argument('-sc', '--scales', type=int, default=4, help="number of scales used (default: 4)")
parser.add_argument('-ly', '--layers', type=int, default=3, help="number of layers used (default: 3)")
parser.add_argument('-iw', '--INC_weight', type=float, default=0.05, help="weight given by loss function to INC layers (default: 0.05)")
parser.add_argument('--visu',  action='store_true', help='show intermediate results')
parser.add_argument('--save',  action='store_true', help='save intermediate results in /tmp folder')
parser.add_argument('--keops', action='store_true', help='use keops package')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('selected device is '+str(device))
args.device = device

synth_img = GotexInceptionV3(args)

# plot and save the synthesized texture 
all_functions.imshow(synth_img)
all_functions.imsave('synthesized.png', synth_img)