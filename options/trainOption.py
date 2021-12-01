import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/haokang/dataset/paired/stylegan_photo2line')
    parser.add_argument('--sample', type=str, default='./sample')
    parser.add_argument('--pretrained_checkpoint', type=str, default='./pretrained_checkpoint')
    parser.add_argument('--output', type=str, default='./output')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--save_img_epoch', type=int, default=1)
    parser.add_argument('--save_model_epoch', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.5)
    parser.add_argument('--lambda_feat', type=float, default=10.0)

    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsampling_G', type=int, default=4)
    parser.add_argument('--n_blocks_G', type=int, default=9)

    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n_layers_D', type=int, default=3)
    parser.add_argument('--use_sigmoid', type=bool, default=False)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--getIntermFeat', type=bool, default=False)

    args = parser.parse_args()

    return args