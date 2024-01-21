import argparse
import torch
import utils as gu
from wgenpatex import GotexVgg
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('target_image_path', help='paths of target texture image')
parser.add_argument('-s', '--size', default=None, help="size of synthetized texture [nrow, ncol] (default: target texture size)")
parser.add_argument('-w', '--patch_size', type=int,default=4, help="patch size (default: 4)")
parser.add_argument('-nmax', '--iter_max', type=int, default=1000, help="max iterations of the algorithm(default: 1000)")
parser.add_argument('-npsi', '--iter_psi', type=int, default=20, help="max iterations for psi (default: 20)")
parser.add_argument('-ilr', '--img_lr', type=float, default=1., help="learning rate for the LBFGS image optimizer (default: 1.)")
parser.add_argument('-plr', '--psi_lr', type=float, default=1., help="learning rate for the SGD psi optimizer (default: 1.)")
parser.add_argument('-nout', '--n_patches_out', type=int, default=2000, help="number maximum of patches of the target texture used, -1 corresponds to all patches (default: 2000)")
parser.add_argument('-sc', '--scales', type=int, default=4, help="number of scales used (default: 4)")
parser.add_argument('-ly', '--layers', type=int, default=5, help="number of layers used (default: 5)")
parser.add_argument('--visu',  action='store_true', help='show intermediate results')
parser.add_argument('--save',  action='store_true', help='save intermediate results in /tmp folder')
parser.add_argument('--keops', action='store_true', help='use keops package')
args = parser.parse_args()

if __name__ == "__main__":
     
    # select cuda device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('selected device is '+str(device))
    args.device = device
    
    # run synthesis
    synth_img, loss_list = GotexVgg(args)
    
    # plot and save the synthesized texture 
    gu.ShowImg(gu.PostProc(synth_img))
    plt.show()
    gu.SaveImg('synthesized.png', gu.PostProc(synth_img.clone().detach()))
    plt.plot(loss_list)
    plt.savefig('loss.png')
    plt.show()