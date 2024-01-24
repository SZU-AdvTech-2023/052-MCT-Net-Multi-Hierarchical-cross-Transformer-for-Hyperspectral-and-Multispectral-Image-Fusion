import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from models.MCT import MCT
from models.model import my_model
from utils import *
from data_loader import build_datasets
from dataloader import CAVE_train_dhif
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F
import yaml
from myvalidate import myvalidate
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR

args = args_parser.args_parser()  # 路径等等一些基础设置,注意默认设置
# torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    print(args)
    # 读取配置文件
    with open('my_config_file.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # 获取数据集配置
    train_dataset_config = config.get('train_dataset', {})
    val_dataset_config = config.get('val_dataset', {})

    # 可以创建数据集实例了
    # 为训练数据集创建一个实例
    train_dataset = CAVE_train_dhif(
        root=train_dataset_config['dataset']['args']['root'],
        istrain=train_dataset_config['dataset']['args']['istrain'],
        upscale_factor=train_dataset_config['dataset']['args']['upscale_factor'],
        sizeI=train_dataset_config['dataset']['args']['sizeI'],
        trainset_num=train_dataset_config['dataset']['args']['trainset_num']
    )

    # 为验证数据集创建一个实例
    val_dataset = CAVE_train_dhif(
        root=val_dataset_config['dataset']['args']['root'],
        istrain=val_dataset_config['dataset']['args']['istrain'],
        upscale_factor=val_dataset_config['dataset']['args']['upscale_factor'],
        sizeI=val_dataset_config['dataset']['args']['sizeI']
    )

    # Custom dataloader
    # 自定义数据加载器
    if args.dataset == 'CAVE_train_dhif':
        train_loader = DataLoader(dataset=train_dataset, batch_size=train_dataset_config['batch_size'], shuffle=True,
                                  num_workers=4)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True,
                                num_workers=4)
    else:
        train_list, test_list = build_datasets(args.root, args.dataset, args.image_size, args.n_select_bands,
                                               args.scale_ratio)
    # 默认的root为“./data”，dataset为“Washington”，scale_ratio为“4”，n_select_bands为“5”，image_size为“128”，通过这些加载训练集和测试集
    # args.n_bands为输出的通道数，默认为0，但下面的选择会根据数据集去对应选择输出的通道数
    if args.dataset == 'PaviaU':
        args.n_bands = 103
    elif args.dataset == 'Pavia':
        args.n_bands = 102
    elif args.dataset == 'Botswana':
        args.n_bands = 145
    elif args.dataset == 'KSC':
        args.n_bands = 176
    elif args.dataset == 'Urban':
        args.n_bands = 162
    elif args.dataset == 'IndianP':
        args.n_bands = 200
    elif args.dataset == 'Washington':
        args.n_bands = 191
    elif args.dataset == 'MUUFL_HSI':
        args.n_bands = 64
    elif args.dataset == 'Salinas_corrected':
        args.n_bands = 204
    elif args.dataset == 'Houston_HSI':
        args.n_bands = 144
    elif args.dataset == 'CAVE_train_dhif':
        args.n_bands = 32
    # Build the models
    if args.arch == 'SSFCNN':
        model = SSFCNN(args.scale_ratio, args.n_select_bands, args.n_bands).cuda()
    elif args.arch == 'ConSSFCNN':
        model = ConSSFCNN(args.scale_ratio, args.n_select_bands, args.n_bands).cuda()
    elif args.arch == 'TFNet':
        model = TFNet(args.scale_ratio, args.n_select_bands, args.n_bands).cuda()
    elif args.arch == 'ResTFNet':
        model = ResTFNet(args.scale_ratio, args.n_select_bands, args.n_bands).cuda()
    elif args.arch == 'MSDCNN':
        model = MSDCNN(args.scale_ratio, args.n_select_bands, args.n_bands).cuda()
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
        model = SSRNET(args.arch, args.scale_ratio, args.n_select_bands, args.n_bands, ).cuda()
    elif args.arch == 'SpatCNN':
        model = SpatCNN(args.scale_ratio, args.n_select_bands, args.n_bands).cuda()
    elif args.arch == 'SpecCNN':
        model = SpecCNN(args.scale_ratio, args.n_select_bands, args.n_bands).cuda()
    elif args.arch == 'MCT':
        model = MCT(args.arch, args.scale_ratio, args.n_select_bands, args.n_bands, args.dataset).cuda()
    elif args.arch == 'my_model':
        model = my_model(Ch=31, stages=4, sf=train_dataset.factor)
        # 通道数是否要写死？或者写成train_dataset.HR_HSI[0,0,:,0].size?
        model = model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.9999))  # 参数迭代优化，学习率为lr
    lr_scheduler = MultiStepLR(optimizer, gamma=0.95,
                               milestones=[1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96,
                                           101, 106, 111, 116, 121, 126, 131, 136, 141, 146])
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + 'M')
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
        .replace('arch', args.arch)
    # model_path中默认的字符串的相关部分替换成数据集名称与模型名称
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        # try:
        #     model.load_state_dict(torch.load("./checkpoints/epoch-best.pth")['model']['sd'])
        # except:
        #     model = nn.DataParallel(model).cuda()
        #     model.load_state_dict(torch.load("./checkpoints/epoch-best.pth")['model']['sd'])
        # print('Load the chekpoint of {}'.format(model_path))
        # recent_psnr = validate(test_list,args.arch,args.dataset,model,0,args.n_epochs)
        # PSNR是峰值信噪比，SSIM为相似性结构，
        # print ('psnr: ', recent_psnr)

    # Loss and Optimizer
    if args.arch == 'my_model':
        criterion = nn.L1Loss().cuda()
    else :
        criterion = nn.MSELoss().cuda()
    # 均方差的损失函数

    best_psnr = 0
    best_ssim = 0
    # best_psnr = validate(test_list,args.arch,args.dataset,model,0,args.n_epochs)
    # print ('psnr: ', best_psnr)

    # Epochs
    print('Start Training: ')
    best_epoch = 0
    if args.arch == 'my_model':
        # num_epochs=train_dataset.num
        for epoch in range(args.n_epochs):
            model.train()
            running_loss = 0.0
            for hr_msi, lr_hsi, hr_hsi in train_loader:
                hr_msi, lr_hsi, hr_hsi = hr_msi.to(device), lr_hsi.to(device), hr_hsi.to(device)
                optimizer.zero_grad()
                outputs = model(hr_msi, lr_hsi)
                loss = criterion(outputs, hr_hsi)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            lr_scheduler.step()
            print(f"Lr: {optimizer.param_groups[0]['lr']}, Epoch {epoch + 1}/{args.n_epochs}, Loss: {running_loss / len(train_loader)}")
            val_loss, val_psnr = myvalidate(model, val_loader, criterion)
            print(f'Epoch [{epoch + 1}/{args.n_epochs}], Validation Loss: {val_loss}, Validation PSNR: {val_psnr}')
            # 如果PSNR有提升，则保存模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved at epoch {epoch + 1} with PSNR: {best_psnr}")
        print(f"Best PSNR obtained at epoch {best_epoch + 1} with PSNR: {best_psnr}")
    else:
        for epoch in range(args.n_epochs):
            # One epoch's traininginceptionv3
            print('Train_Epoch_{}: '.format(epoch))
            train(train_list, args.image_size, args.scale_ratio, args.n_bands, args.arch, model, optimizer, criterion,
                  epoch, args.n_epochs)

            # optimizer为优化器，criterion为损失函数
            # One epoch's validation
            print('Val_Epoch_{}: '.format(epoch))
            recent_psnr,recent_ssim = validate(test_list, args.arch, args.dataset, model, epoch, args.n_epochs)
            print('psnr: ', recent_psnr)
            print('   ssim: ', recent_ssim)

            # # save model
            is_best = recent_psnr > best_psnr
            best_psnr = max(recent_psnr, best_psnr)
            is_best_ssim = recent_ssim > best_ssim
            best_ssim = max(recent_ssim, best_ssim)
            # if epoch > 9000 and epoch % 50 == 0:
            #     model_path_ = model_path.split('.pkl')[0] + 'ep' + str(epoch) + '.pkl'
            #     print(model_path_)
            #     torch.save(model.state_dict(), model_path_)
            if is_best:
                best_epoch = epoch
                if best_psnr > 0:
                    torch.save(model.state_dict(), model_path)  # 将结果最好的那个轮次的参数保存下来
                    print('Saved!')
                    print('')
            print('best psnr:', best_psnr, 'at epoch:', best_epoch)
            if is_best_ssim:
                print(',ssim is best too:', best_ssim)

    print('best_psnr: ', best_psnr)
    print('   best_ssim: ', best_ssim)
    with open(args.arch + '-' + args.dataset + '.txt', 'a') as f:
        f.write('best epoch:' + str(best_epoch) + '   best psnr:' + str(best_psnr) + '   best ssim:' + str(best_ssim) + '\n')
        f.close()


if __name__ == '__main__':
    main(args)
