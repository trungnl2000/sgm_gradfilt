# Cityscapes->VOC12Aug, Train last 5 layers in DeepLabV3-ResNet18 with svd
python train.py configs/upernet/0.8/svd_5L_upernet_r18_512x512_20k_voc12aug.py --load-from calib/calib_upernet_r18_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233
# Cityscapes->VOC12Aug, Train last 10 layers in DeepLabV3-ResNet18 with svd
python train.py configs/upernet/0.8/svd_10L_upernet_r18_512x512_20k_voc12aug.py --load-from calib/calib_upernet_r18_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233