import torch
from torch.utils.data import DataLoader
from torchvision import utils
import os

from model.Generator import GlobalGenerator
from dataset import TestImageDataSet
from utils.utils import get_norm_layer as get_norm_layer
from options.testOption import get_args

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    device = 'cuda:0'

    G = GlobalGenerator(input_nc=args.input_nc,
                        output_nc=args.output_nc,
                        ngf=args.ngf,
                        n_downsampling=args.n_downsampling_G,
                        n_blocks=args.n_blocks_G,
                        norm_layer=get_norm_layer(norm_type=args.norm)).to(device)
    dataset = TestImageDataSet(data_root=args.test_dir,
                               img_size=args.img_size)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4)

    checkpoint = torch.load(args.checkpoint)
    G.load_state_dict(checkpoint['g'])

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            img = data['img'].to(device)
            fake_img = G(img)

            cat_img = torch.cat([img, fake_img], dim=3)
            save_dir = os.path.join(args.save_dir, str(idx) + '.png')
            utils.save_image(
                cat_img,
                save_dir,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
