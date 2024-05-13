_base_ = "./full_pspnet_mv2-d8_512x512_20k_voc12aug.py"

freeze_layers = [
    "backbone", "~backbone.layer7",
    "~backbone.layer6.2.conv.2", "~backbone.layer6.2.conv.1"
]

gradient_filter = dict(
    enable=True,
    filter_install=[
        dict(path="decode_head.bottleneck", type='cbr', radius=4),
        dict(path="backbone.layer7.0.conv.2", type='cbr', radius=4),
        dict(path="backbone.layer7.0.conv.1", type='cbr', radius=4),
        dict(path="backbone.layer7.0.conv.0", type='cbr', radius=4),
        dict(path="backbone.layer6.2.conv.2", type='cbr', radius=4),
        dict(path="backbone.layer6.2.conv.1", type='cbr', radius=4),
    ]
)
