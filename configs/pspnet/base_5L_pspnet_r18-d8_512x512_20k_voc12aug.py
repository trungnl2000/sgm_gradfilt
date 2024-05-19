_base_ = "./full_pspnet_r18-d8_512x512_20k_voc12aug.py"

freeze_layers = [
    "backbone", 
]


base = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr'),
    ]
)
