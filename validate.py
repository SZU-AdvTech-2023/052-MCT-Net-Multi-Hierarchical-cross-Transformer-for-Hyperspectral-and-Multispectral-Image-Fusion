from torch import nn
from utils import *
import cv2
import pdb
from torchvision.utils import save_image
from PIL import Image

from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam,calc_ssim_hsi_4d,SSIM


def validate(test_list, arch,dataset, model, epoch, n_epochs):
    test_ref, test_lr, test_hr = test_list
    model.eval()

    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = to_var(test_ref).detach()
        lr = to_var(test_lr).detach()
        hr = to_var(test_hr).detach()
        if arch == 'SSRNet':
            out, _, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpat':
            _, out, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpec':
            _, _, out, _, _, _ = model(lr, hr)
        else:
            out, _, _, _, _, _ = model(lr, hr)

        ref = ref.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out)
        sam = calc_sam(ref, out)
        ssim= calc_ssim_hsi_4d(ref,out)
        #ssim2=SSIM(ref,out)
        #myinput=torch.stack((ref[0,1,:,:],ref[0,2,:,:],ref[0,3,:,:]))
        #myoutput=torch.stack((out[0,1,:,:],out[0,2,:,:],out[0,3,:,:]))
        #save_image(myinput, 'in.png')
        #save_image(myoutput, 'out.png')
        # arrayin = myinput.numpy()
        # arrayout=myoutput.numpy()
        # matin = np.uint8(arrayin)
        # matout = np.uint8(arrayout)
        tempin=ref[0,1:4,:,:]
        tempin=tempin.transpose(1, 2, 0)
        tempout = out[0, 1:4, :, :]
        tempout = tempout.transpose(1, 2, 0)
        imin = Image.fromarray(np.uint8(tempin))
        imout =Image.fromarray(np.uint8(tempout))

        with open(arch+'-'+dataset+'.txt', 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) +',' + str(ssim) + '\n')

    return psnr,ssim