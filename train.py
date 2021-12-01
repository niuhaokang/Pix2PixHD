import argparse
import itertools
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import os

from dataset import ImageDataSet
from dataset import TestImageDataSet
from model.Generator import GlobalGenerator
from model.Discriminator import MultiscaleDiscriminator
from utils import get_norm_layer as get_norm_layer
from utils import weights_init as weights_init
from Loss import GANLoss
from Loss import VGGLoss
from options.trainOption import get_args as get_args

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    # 获取相关参数
    args = get_args()

    # 生成结果存放的目录及扩展args
    if not os.path.exists(args.pretrained_checkpoint):
        os.mkdir(args.pretrained_checkpoint)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
        os.mkdir(os.path.join(args.output, 'checkpoints'))
        os.mkdir(os.path.join(args.output, 'imgs'))
        os.mkdir(os.path.join(args.output, 'train_data'))
        os.mkdir(os.path.join(args.output, 'loss_txt'))
    args.checkpoints = os.path.join(args.output, 'checkpoints')
    args.imgs = os.path.join(args.output, 'imgs')
    args.train_data = os.path.join(args.output, 'train_data')
    args.loss_txt = os.path.join(args.output, 'loss_txt')

    # 定义使用gpu及其cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    device = 'cuda:0'

    # 定义dataloader
    dataset = ImageDataSet(data_root=args.data_root,
                           img_size=args.img_size)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_dataset = TestImageDataSet(data_root=args.sample,
                                    img_size=args.img_size)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                shuffle=False,
                                num_workers=4)

    # 定义模型
    G = GlobalGenerator(input_nc=args.input_nc,
                        output_nc=args.output_nc,
                        ngf=args.ngf,
                        n_downsampling=args.n_downsampling_G,
                        n_blocks=args.n_blocks_G,
                        norm_layer=get_norm_layer(norm_type=args.norm)).to(device)
    G.apply(weights_init)

    D = MultiscaleDiscriminator(input_nc=args.input_nc,
                                 ndf=args.ndf,
                                 n_layers=args.n_layers_D,
                                 norm_layer=get_norm_layer(norm_type=args.norm),
                                 use_sigmoid=args.use_sigmoid,
                                 num_D=args.num_D,
                                 getIntermFeat=args.getIntermFeat).to(device)
    D.apply(weights_init)

    # 加载预训练模型
    pretrained_checkpoint = os.path.join(args.pretrained_checkpoint, 'latent.pt')
    if os.path.exists(pretrained_checkpoint):
        print("载入预训练模型: {}".format(pretrained_checkpoint))
        G.load_state_dict(torch.load(pretrained_checkpoint)['g'])
        D.load_state_dict(torch.load(pretrained_checkpoint)['d'])

    # 定义损失函数
    criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor, device=device)
    criterionFeat = torch.nn.L1Loss()
    criterionVGG = VGGLoss(device=device)

    # 定义优化器
    G_optim = torch.optim.Adam(G.parameters(),
                               lr=args.lr,
                               betas=(args.beta1, args.beta2))
    D_optim = torch.optim.Adam(D.parameters(),
                               lr=args.lr,
                               betas=(args.beta1, args.beta2))

    # 定义训练过程
    for epoch in range(args.epoch):
        # 保存测试样例
        if epoch % args.save_img_epoch == 0:
            with torch.no_grad():
                for idx, test_data in enumerate(test_dataloader):
                    this_epoch = os.path.join(args.imgs, 'epoch_' + str(epoch))
                    if not os.path.exists(this_epoch):
                        os.mkdir(this_epoch)

                    name = str(idx) + '.png'
                    img = test_data['img'].to(device)
                    fake_img = G(img)

                    target_img = torch.cat([img, fake_img], dim=3)
                    target_dir = os.path.join(this_epoch, name)
                    utils.save_image(
                        target_img,
                        target_dir,
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )

        # 保存模型
        if epoch % args.save_model_epoch == 0:
            torch.save({
                "g": G.state_dict(),
                "d": D.state_dict()
            }, os.path.join(args.checkpoints, 'epoch_' + str(epoch) + '.pt'))
            torch.save({
                "g": G.state_dict(),
                "d": D.state_dict()
            }, os.path.join(args.pretrained_checkpoint, 'latent.pt'))

        # 迭代训练
        for idx, data in enumerate(dataloader):
            # 记录所有损失
            loss_dict = {}

            # 输入数据
            imgA = data['A'].to(device)
            imgB = data['B'].to(device)

            imgA2B = G(imgA)

            # 更新D
            requires_grad(G, False)
            requires_grad(D, True)

            real_pred = D(imgB)
            fake_pred = D(imgA2B)

            D_Adv = 1/2 * criterionGAN(fake_pred, False) +\
                     1/2 * criterionGAN(real_pred, True)
            loss_dict['D_Adv'] = D_Adv

            D_loss = D_Adv

            D.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optim.step()

            # 更新G
            requires_grad(G, True)
            requires_grad(D, False)

            real_pred = D(imgB)
            fake_pred = D(imgA2B)

            G_loss_adv = criterionGAN(fake_pred, True)

            GAN_Feat = 0.0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            weights = 1.0 / args.num_D
            for i in range(args.num_D):
                for j in range(len(fake_pred[i])):
                    GAN_Feat += weights * feat_weights * \
                                       criterionFeat(fake_pred[i][j],
                                                     real_pred[i][j].detach()) * args.lambda_feat

            loss_VGG = criterionVGG(imgA2B, imgB) * args.lambda_feat

            loss_dict['G_Adv'] = G_loss_adv
            loss_dict['Feat'] = GAN_Feat
            loss_dict['VGG'] = loss_VGG

            G_loss = G_loss_adv + GAN_Feat + loss_VGG

            G.zero_grad()
            G_loss.backward()
            G_optim.step()

            # 打印记录本个batch的损失
            loss_log = ''
            for l in loss_dict:
                loss_log += l + ":" + str(round(loss_dict[l].item(), 4)) + '; '
            print(('epoch_{}_{}:' + loss_log).format(epoch,idx))

            epoch_loss_dir = os.path.join(args.loss_txt, 'epoch_' + str(epoch) + '.txt')
            with open(epoch_loss_dir, 'a+') as f:
                f.write(loss_log + '\n')

            # 保存本轮训练样例
            if idx >= 0 and idx < 10:
                this_epoch = os.path.join(args.train_data, 'epoch_' + str(epoch))
                if not os.path.exists(this_epoch):
                    os.mkdir(this_epoch)

                imgs = torch.cat([imgA, imgB, imgA2B], dim=3)

                utils.save_image(
                    imgs,
                    os.path.join(this_epoch, str(idx) + '.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )


