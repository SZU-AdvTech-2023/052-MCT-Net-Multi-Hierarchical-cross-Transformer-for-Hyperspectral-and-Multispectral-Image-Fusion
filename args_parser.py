import argparse

def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='MCT',
                        choices=[  # these four models are used for ablation experiments
                            #这四个模型用于  实验
                            'SpatCNN', 'SpecCNN',
                            'SpatRNET', 'SpecRNET',
                            # the proposed method
                            #推荐使用的模型
                            'SSRNET','MCT',
                            # these five models are used for comparison experiments
                            #这五个模型用于比较实验
                            'SSFCNN', 'ConSSFCNN',
                            'TFNet', 'ResTFNet',
                            'MSDCNN','my_model'
                        ])

    parser.add_argument('-root', type=str, default='./data')
    parser.add_argument('-dataset', type=str, default='Pavia',
                        choices=['PaviaU', 'Botswana', 'KSC', 'Urban', 'Pavia', 'IndianP', 'Washington','MUUFL_HSI','Salinas_corrected','Houston_HSI','CAVE_train_dhif'])
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=0)
    parser.add_argument('--n_select_bands', type=int, default=5)

    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/dataset_arch.pkl',
                        help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',
                        help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',
                        help='directory for resized images')

    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=3e-4)#默认的学习率
    parser.add_argument('--image_size', type=int, default=128)

    args = parser.parse_args()
    return args
