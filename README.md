Install requirement:
```
pip install scipy
pip install gdown
```
Fix lỗi "TypeError: FormatCode() got an unexpected keyword argument 'verify'"

```
pip install yapf==0.40.1
```

Prepare dataset:

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar xvf benchmark.tgz
rm benchmark.tgz
mkdir mmsegmentation/data
mkdir VOCdevkit/VOCaug
mv benchmark_RELEASE/dataset VOCdevkit/VOCaug
mv VOCdevkit mmsegmentation/data
cd mmsegmentation
python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
mv data ./../
cd ..
rm -r benchmark_RELEASE
```
Prepare checkpoint
```
gdown https://drive.google.com/uc?id=1CxLWBcByKFS-Vv75VG70DWUX3EcFmSu1
tar xvf seg_calib_ckpt.tar.gz
rm seg_calib_ckpt.tar.gz
```

