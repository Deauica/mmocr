# buffer.py

stage = 1

if stage == 1:
    from mmengine import Config

    cfg_path = "configs/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py"
    cfg = Config.fromfile(cfg_path)
    print(cfg)

