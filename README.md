# 关于复现工作

我选择复现的论文是 **高光谱与多光谱融合超分的多头注意力机制**. 它的英文名称是 **MCT-Net: Multi-hierarchical cross transformer for hyperspectral and multispectral image fusion** on Knowledge-Based Systems 264 (2023) 110362

# 高光谱与多光谱融合超分的多头注意力机制 (MCT-Net) 复现项目

## 项目简介

本项目是对论文《高光谱与多光谱融合超分的多头注意力机制》进行的复现工作。该论文提出了一种名为 MCT-Net 的多头注意力机制，用于高光谱与多光谱图像的融合超分辨率。

## 环境配置

### 硬件环境
- CPU: Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz
- GPU: NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7
- 内存: 256GB

### 软件环境
- 操作系统: Ubuntu 20.04.3 LTS
- 编程语言: Python 3.9

### 依赖项和库
确保在搭建环境前安装以下依赖项：
```bash
pip install -r requirements.txt

# 关于代码目录

F:.
│  args_parser.py
│  data_loader.py
│  dataloader.py
│  main.py
│  metrics.py
│  test.py
│  train.py
│  utils.py
│  validate.py
│  list.txt
│  
├─.idea
│  │  .gitignore
│  │  deployment.xml
│  │  MCT-Net.iml
│  │  misc.xml
│  │  modules.xml
│  │  vcs.xml
│  │  workspace.xml
│  │  
│  └─inspectionProfiles
│          profiles_settings.xml
│          
├─checkpoints
│      Botswana_MCT.pkl
│      Botswana_MSDCNN.pkl
│      Botswana_ResTFNet.pkl
│      Botswana_SSRNET.pkl
│      CAVE_train_dhif_my_model.pkl
│      Houston_HSI_MCT.pkl
│      Houston_HSI_MSDCNN.pkl
│      Houston_HSI_ResTFNet.pkl
│      Houston_HSI_SSRNET.pkl
│      IndianP_MCT.pkl
│      IndianP_MSDCNN.pkl
│      IndianP_ResTFNet.pkl
│      IndianP_SSRNET.pkl
│      KSC_MCT.pkl
│      KSC_MSDCNN.pkl
│      KSC_ResTFNet.pkl
│      KSC_SSRNET.pkl
│      MUUFL_HSI_MCT.pkl
│      MUUFL_HSI_MSDCNN.pkl
│      MUUFL_HSI_ResTFNet.pkl
│      MUUFL_HSI_SSRNET.pkl
│      Pavia_MCT.pkl
│      Pavia_MSDCNN.pkl
│      Pavia_ResTFNet.pkl
│      Pavia_SSRNET.pkl
│      PaviaU_MCT.pkl
│      PaviaU_MSDCNN.pkl
│      PaviaU_ResTFNet.pkl
│      PaviaU_SSRNET.pkl
│      Salinas_corrected_MCT.pkl
│      Salinas_corrected_MSDCNN.pkl
│      Salinas_corrected_ResTFNet.pkl
│      Salinas_corrected_SSRNET.pkl
│      Urban_MCT.pkl
│      Urban_MSDCNN.pkl
│      Urban_ResTFNet.pkl
│      Urban_SSRNET.pkl
│      Urban_TFNet.pkl
│      Washington_MCT.pkl
│      Washington_MSDCNN.pkl
│      Washington_ResTFNet.pkl
│      Washington_SSRNET.pkl
│      
├─figs
│      Botswana_lr.jpg
│      Botswana_lr_dif.jpg
│      Botswana_MCT_out.jpg
│      Botswana_MCT_out_dif.jpg
│      Botswana_ref.jpg
│      Houston_HSI_lr.jpg
│      Houston_HSI_lr_dif.jpg
│      Houston_HSI_MCT_out.jpg
│      Houston_HSI_MCT_out_dif.jpg
│      Houston_HSI_ref.jpg
│      KSC_lr.jpg
│      KSC_lr_dif.jpg
│      KSC_MCT_out.jpg
│      KSC_MCT_out_dif.jpg
│      KSC_ref.jpg
│      Pavia_lr.jpg
│      Pavia_lr_dif.jpg
│      Pavia_MCT_out.jpg
│      Pavia_MCT_out_dif.jpg
│      Pavia_ref.jpg
│      PaviaU_lr.jpg
│      PaviaU_lr_dif.jpg
│      PaviaU_MCT_out.jpg
│      PaviaU_MCT_out_dif.jpg
│      PaviaU_ref.jpg
│      Washington_lr.jpg
│      Washington_lr_dif.jpg
│      Washington_MCT_out.jpg
│      Washington_MCT_out_dif.jpg
│      Washington_ref.jpg
│      
├─figure
│      CiT_Net.jpg
│      
└─models
    │  basic_blocks.py
    │  IntmdSequential.py
    │  MCT.py
    │  model.py
    │  MSDCNN.py
    │  scalablevit.py
    │  SingleCNN.py
    │  SSFCNN.py
    │  SSRNET.py
    │  TFNet.py
    │  Transformer.py
    │  
    └─sync_batchnorm
            __init__.py
            batchnorm.py
            batchnorm_reimpl.py
            comm.py
            model.py
            replicate.py
            unittest.py
 ```



## 数据集介绍

本项目使用了以下高光谱图像数据集：

- **Pavia Center**
- **Pavia University**
- **Urban**
- **Botswana**
- **Washington DC Mall**
- **Indian Pines (新添加)**
- **MUUFL Gulfport Hyperspectral Dataset (MUUFL_HSI) (新添加)**
- **Salinas Corrected (新添加)**
- **Houston Hyperspectral Image (Houston_HSI) (新添加)**

## 复现细节

### 与已有开源代码对比

- 在原文代码的基础上，完成了相关环境的部署，并成功复现了文章的实验结果。
- 在原文使用的数据集的基础上，新添加了四个数据集，并完成适配工作，使模型在新的数据集上得到验证的结果。
- 添加了新的评价指标 SSIM，使得文章的评价体系更完善和多方面。

### 实验环境搭建

1. **克隆仓库**


2. **安装依赖项**

    ```bash
    pip install -r requirements.txt
    ```

3. **配置说明**

    修改 `myconfig.yaml` 文件以适应你的实验设置。

4. **运行实验**

    ```bash
    python main.py
    python test.py
    ```

### 结果分析

在 Pavia Center 数据集上，我们的方法获得了最低的 RMSE、最高的 PSNR，最低的 ERGAS 和 SAM。这表明我们的方法在光谱信息保留、空间细节还原、整体相对误差和光谱保真度等方面表现出色。

## 创新点

引入结构相似性指数 (SSIM) 评价指标是本项目的创新点。SSIM 综合考虑了亮度、对比度和结构，提供了更为全面、结构感知的图像质量度量方法。

## 使用说明

如果您希望运行实验或测试您的模型，可以按照上述环境配置和运行实验的步骤进行。请确保您已经安装了所有依赖项，并配置好实验所需的参数。

## License

MIT License

Copyright (c) [2024] [cyx]


## 其他的

后期不再单独维护，请不必提issue。谢谢