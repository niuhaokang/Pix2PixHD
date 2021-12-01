import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='./pretrained_checkpoint/latent.pt')
    parser.add_argument('--test_dir', type=str, default='./sample')
    parser.add_argument('--save_dir', type=str, default='./save')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)

    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsampling_G', type=int, default=4)
    parser.add_argument('--n_blocks_G', type=int, default=9)

    args = parser.parse_args()

    return args