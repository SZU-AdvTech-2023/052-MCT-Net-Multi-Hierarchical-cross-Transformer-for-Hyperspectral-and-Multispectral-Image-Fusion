import torch
from torch.utils.data import Dataset
import scipy.io as scio
import os
import random
import numpy as np
import cv2


def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img


def psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    # Compute the OTF
    otf = np.fft.fft2(psf)
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def para_setting(kernel_type, sf, sz, sigma):
    if kernel_type == 'uniform_blur':
        psf = np.ones([sf, sf]) / (sf * sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B, fft_BT


class CAVE_train_dhif(Dataset):
    def __init__(self, root, istrain, upscale_factor, sizeI, trainset_num=20):
        super(CAVE_train_dhif, self).__init__()
        files = os.listdir(root)
        files.sort()
        HR_HSI = np.zeros((512, 512, 31, len(files)))
        HR_MSI = np.zeros((512, 512, 3, len(files)))
        for idx in range(len(files)):
            data = scio.loadmat(os.path.join(root, files[idx]))
            HR_HSI[:, :, :, idx] = data['hr'].transpose(1, 2, 0)
            HR_MSI[:, :, :, idx] = data['rgb']
        self.istrain = istrain
        self.factor = upscale_factor
        if istrain:
            self.num = trainset_num
            self.file_num = len(files)
            self.sizeI = sizeI
        else:
            self.num = len(files)
            self.file_num = len(files)
            self.sizeI = sizeI
        self.HR_HSI, self.HR_MSI = HR_HSI, HR_MSI

    def H_z(self, z, factor, fft_B):
        f = torch.fft.fft2(z, dim=(-2, -1))  # [1, 31, 96, 96]
        f = torch.stack((f.real, f.imag), -1)
        # -------------------complex myltiply-----------------#
        if len(z.shape) == 3:
            ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
            M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                           (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
            Hz = torch.irfft(M, 2, onesided=False)
            x = Hz[:, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
            M = torch.cat(
                ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                 (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
            Hz = torch.fft.ifft2(torch.complex(M[..., 0], M[..., 1]), dim=(-2, -1)).real
            x = Hz[:, :, int(factor // 2) - 1::factor, int(factor // 2) - 1::factor]
        return x

    def __getitem__(self, index):
        if self.istrain == True:
            index1 = random.randint(0, self.file_num - 1)
        else:
            index1 = index

        sigma = 2.0
        HR_HSI = self.HR_HSI[:, :, :, index1]
        HR_MSI = self.HR_MSI[:, :, :, index1]

        sz = [self.sizeI, self.sizeI]
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)), 2)

        px = random.randint(0, 512 - self.sizeI)
        py = random.randint(0, 512 - self.sizeI)
        hr_hsi = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hr_msi = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

        if self.istrain == True:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hr_hsi = np.rot90(hr_hsi)
                hr_msi = np.rot90(hr_msi)

            # Random vertical Flip
            for j in range(vFlip):
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2, 0, 1).unsqueeze(0)
        hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
        lr_hsi = torch.FloatTensor(lr_hsi)

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)

        return hr_msi, lr_hsi, hr_hsi

    def __len__(self):
        return self.num
