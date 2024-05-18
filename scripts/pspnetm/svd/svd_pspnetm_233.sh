# Cityscapes->VOC12Aug, Train last 5 layers in DeepLabV3-ResNet18 with svd
python train.py configs/pspnetmv2/svd_5L_pspnet_mv2-d8_512x512_20k_voc12aug.py --load-from calib/calib_pspnet_mv2-d8_512x512_5k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233
# Cityscapes->VOC12Aug, Train last 10 layers in DeepLabV3-ResNet18 with svd
python train.py configs/pspnetmv2/svd_10L_pspnet_mv2-d8_512x512_20k_voc12aug.py --load-from calib/calib_pspnet_mv2-d8_512x512_5k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233