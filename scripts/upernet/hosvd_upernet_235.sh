# Cityscapes->VOC12Aug, Train last 5 layers in DeepLabV3-ResNet18 with HOSVD
python train.py configs/upernet/hosvd_5L_upernet_r18_512x512_20k_voc12aug.py --load-from calib/calib_upernet_r18_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 235
# Cityscapes->VOC12Aug, Train last 10 layers in DeepLabV3-ResNet18 with HOSVD
python train.py configs/upernet/hosvd_10L_upernet_r18_512x512_20k_voc12aug.py --load-from calib/calib_upernet_r18_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 235