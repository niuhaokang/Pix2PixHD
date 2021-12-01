# Image Translation with Pix2PixHD 

## Introduce 
本项目是基于官方 [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD) 实现的有监督图像翻译

## How to use
### Train
1. 将两个数据域的图像以如下结构部署
```text
---data_root
------A
------B
``` 
2. 训练
```shell
# example
python train.py --data_root xxx --sample ./sample --img_size 256
# 更多参数指定详情请看./options/trainOption.py
```
### Test
1. 测试
```shell
python train.py --test_dir xxx --save_dir xxx --img_size 256
```

### Project Struct 
* model: 存放神经网络机构
* option: 用于定义超参数
* output: 存放训练过程的中间结果
* pretrained_checkpoint: 存放预训练模型

### Acknowledge
This code borrows heavily from [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD).
