python train.py configs/deeplabv3mv2/full_deeplabv3_mv2_512x512_20k_voc12aug.py --load-from calib/calib_deeplabv3_mv2_512x512_5k_voc12aug_cityscapes/version_0/latest.pth --cfg-options data.samples_per_gpu=8 --seed 233