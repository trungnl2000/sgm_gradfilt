import logging
from functools import reduce
import torch.nn as nn
from .conv_avg import wrap_conv_layer
from .conv_hosvd_with_var import wrap_convHOSVD_with_var_layer
from .conv_svd_with_var import wrap_convSVD_with_var_layer
from utils import attach_hooks_for_conv

DEFAULT_CFG = {
    "path": "",
    "radius": 8,
    "type": "",
    "SVD_var": 0.8,
}


def add_grad_filter(module: nn.Module, cfg, hook):
    if cfg['type'] == 'cbr':
        module.conv = wrap_conv_layer(module.conv, cfg['radius'], True)

        attach_hooks_for_conv(module=module.conv, name=cfg['path']+'.conv', hook=hook, special_param=cfg['radius'])
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_conv_layer(module.conv1, cfg['radius'], True)
        module.conv2 = wrap_conv_layer(module.conv2, cfg['radius'], True)

        attach_hooks_for_conv(module=module.conv1, name=cfg['path']+'.conv1', hook=hook, special_param=cfg['radius'])
        attach_hooks_for_conv(module=module.conv2, name=cfg['path']+'.conv2', hook=hook, special_param=cfg['radius'])
    elif cfg['type'] == 'conv':
        module = wrap_conv_layer(module, cfg['radius'], True)

        attach_hooks_for_conv(module=module, name=cfg['path'], hook=hook, special_param=cfg['radius'])
    else:
        raise NotImplementedError
    return module

def add_hosvd_with_var_filter(module: nn.Module, cfg, hook):
    if cfg['type'] == 'cbr':
        module.conv = wrap_convHOSVD_with_var_layer(module.conv, cfg['SVD_var'], True)

        attach_hooks_for_conv(module=module.conv, name=cfg['path']+'.conv', hook=hook, special_param=cfg['SVD_var'])
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_convHOSVD_with_var_layer(module.conv1, cfg['SVD_var'], True)
        module.conv2 = wrap_convHOSVD_with_var_layer(module.conv2, cfg['SVD_var'], True)

        attach_hooks_for_conv(module=module.conv1, name=cfg['path']+'.conv1', hook=hook, special_param=cfg['SVD_var'])
        attach_hooks_for_conv(module=module.conv2, name=cfg['path']+'.conv2', hook=hook, special_param=cfg['SVD_var'])
    elif cfg['type'] == 'conv':
        module = wrap_convHOSVD_with_var_layer(module, cfg['SVD_var'], True)

        attach_hooks_for_conv(module=module, name=cfg['path'], hook=hook, special_param=cfg['SVD_var'])
    else:
        raise NotImplementedError
    return module

def add_svd_with_var_filter(module: nn.Module, cfg, hook):
    if cfg['type'] == 'cbr':
        module.conv = wrap_convSVD_with_var_layer(module.conv, cfg['SVD_var'], True)

        attach_hooks_for_conv(module=module.conv, name=cfg['path']+'.conv', hook=hook, special_param=cfg['SVD_var'])
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_convSVD_with_var_layer(module.conv1, cfg['SVD_var'], True)
        module.conv2 = wrap_convSVD_with_var_layer(module.conv2, cfg['SVD_var'], True)

        attach_hooks_for_conv(module=module.conv1, name=cfg['path']+'.conv1', hook=hook, special_param=cfg['SVD_var'])
        attach_hooks_for_conv(module=module.conv2, name=cfg['path']+'.conv2', hook=hook, special_param=cfg['SVD_var'])
    elif cfg['type'] == 'conv':
        module = wrap_convSVD_with_var_layer(module, cfg['SVD_var'], True)

        attach_hooks_for_conv(module=module, name=cfg['path'], hook=hook, special_param=cfg['SVD_var'])
    else:
        raise NotImplementedError
    return module

def add_hook_for_base_conv(module: nn.Module, cfg, hook):
    if cfg['type'] == 'cbr':
        attach_hooks_for_conv(module=module.conv, name=cfg['path']+'.conv', hook=hook)
    elif cfg['type'] == 'resnet_basic_block':
        attach_hooks_for_conv(module=module.conv1, name=cfg['path']+'.conv1', hook=hook)
        attach_hooks_for_conv(module=module.conv2, name=cfg['path']+'.conv2', hook=hook)
    elif cfg['type'] == 'conv':
        attach_hooks_for_conv(module=module, name=cfg['path'], hook=hook)
    else:
        raise NotImplementedError
    return module

###############################################################################
def register_filter(module, cfgs, hook):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_grad_filter(target, cfg, hook)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_HOSVD_with_var(module, cfgs, hook):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_hosvd_with_var_filter(target, cfg, hook)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_SVD_with_var(module, cfgs, hook):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_svd_with_var_filter(target, cfg, hook)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)



def attach_hook_for_base_conv(module, cfgs, hook):
    filter_install_cfgs = cfgs['filter_install']
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_hook_for_base_conv(target, cfg, hook)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)