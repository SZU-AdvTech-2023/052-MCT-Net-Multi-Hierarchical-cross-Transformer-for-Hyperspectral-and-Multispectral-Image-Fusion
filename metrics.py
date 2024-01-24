import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    #MSE为均方误差
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam

def calc_ssim_hsi_4d(img_tgt, img_fus, sigma=1.5, L=1):
    """
    计算两个高光谱图像之间的平均结构相似性指数（HSI-SSIM）。

    参数:
    - img_tgt: 目标高光谱图像（参考图像），四维数组。
    - img_fus: 融合高光谱图像（待评估图像），四维数组。
    - sigma: 高斯滤波器的标准差值（默认为1.5）。
    - L: 图像的动态范围（默认为1）。

    返回:
    - ssim_index: 平均 HSI-SSIM 指数值。
    """

    # 初始化SSIM值
    ssim_values = []

    # 对每个样本计算HSI-SSIM
    for i in range(img_tgt.shape[0]):
        img_tgt_sample = img_tgt[i, :, :, :]
        img_fus_sample = img_fus[i, :, :, :]

        # 初始化样本中每个波段的SSIM值
        ssim_sample = []

        # 对每个波段计算SSIM
        for j in range(img_tgt_sample.shape[0]):
            img_tgt_band = img_tgt_sample[j,:, :]
            img_fus_band = img_fus_sample[j,:, :]

            ssimi=ssim(img_tgt_band, img_fus_band, data_range=img_fus_band.max() - img_fus_band.min())
            ssim_sample.append(ssimi)



        # 将每个波段的SSIM取平均，得到样本的HSI-SSIM值
        ssim_values.append(np.mean(ssim_sample))

    # 取所有样本的HSI-SSIM的平均值
    ssim_index = np.mean(ssim_values)

    return ssim_index

def SSIM(gt, pred, sigma=1.5, L=1):
    """
    计算两个图像之间的结构相似性指数（SSIM）。

    参数:
    - pred: 预测图像。
    - gt: 参考图像。
    - sigma: 高斯滤波器的标准差值（默认为1.5）。
    - L: 图像的动态范围（默认为1）。

    返回:
    - ssim: SSIM值。
    """

    # 确保图像为NumPy数组
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    # SSIM计算所需的常数
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 高斯滤波
    pred_smooth = gaussian_filter(pred, sigma)
    gt_smooth = gaussian_filter(gt, sigma)

    # 协方差项
    cov_pred_gt = gaussian_filter(pred * gt, sigma) - pred_smooth * gt_smooth

    # 均值
    mean_pred = pred_smooth
    mean_gt = gt_smooth

    # SSIM公式
    num_ssim = (2 * mean_pred * mean_gt + C1) * (2 * cov_pred_gt + C2)
    denom_ssim = (mean_pred ** 2 + mean_gt ** 2 + C1) * (pred_smooth ** 2 + gt_smooth ** 2 + C2)

    ssim = np.mean(num_ssim / denom_ssim)

    return ssim