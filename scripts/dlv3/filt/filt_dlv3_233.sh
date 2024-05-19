# Cityscapes->VOC12Aug, Train last 5 layers in DeepLabV3-ResNet18 with filt
python train.py configs/deeplabv3/filt_5L_deeplabv3_r18-d8_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233
# Cityscapes->VOC12Aug, Train last 10 layers in DeepLabV3-ResNet18 with filt
python train.py configs/deeplabv3/filt_10L_deeplabv3_r18-d8_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_r18-d8_512x512_1k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233