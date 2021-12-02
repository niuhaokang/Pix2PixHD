import shutil
import os

def move(name, root='/data/haokang/result'):
    target = os.path.join(root, name)
    os.makedirs(target)

    shutil.move('../output', os.path.join(target, 'output'))
    shutil.move('../pretrained_checkpoint', os.path.join(target, 'pretrained_checkpoint'))


if __name__ == '__main__':
    move('Pix2PixHD_ResGenerator')

