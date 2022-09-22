""" config file main.py """

stage = 6

if stage == 1: 
    # Load use Config.fromfile
    from mmengine.config import Config 

    config_file = "learn_read_config.py"
    cfg = Config.fromfile(config_file)
    print(cfg)

    # 可以直接使用 Dict 或者是 . 来访问对应的属性 
    print("cfg.test_int: ", cfg.test_int)
    print("cfg.test_list: ", cfg.test_list)
    print("cfg.test_dict: ", cfg.test_dict)

if stage == 2: 
    # 可以将 config 与 Registry 结合在一起使用
    from mmengine import Config
    from mmengine.registry import OPTIMIZERS
    from torch import nn 
    
    # specify the key-value pair parameters
    cfg = Config.fromfile("optimizer_cfg.py")
    model = nn.Conv2d(1, 1, 1)
    cfg.optimizer.params = model.parameters()

    # create optimizer use REGISTRY
    optimizer = OPTIMIZERS.build(cfg.optimizer)
    print(optimizer)

if stage == 3: 
    # 配置文件的继承
    from mmengine import Config 
    cfg_file = "resnet50.py"

    cfg = Config.fromfile(cfg_file)
    print(cfg)

if stage == 4: 
    # 配置文件 继承后 稍加修改
    # 需要注意的是，配置文件 继承后的修改，大概可以分成 两个部分： 
    # 1. dict 类型的修改，分别修改 某一个字段，或者是 下级字段就可以了；
    # 2. number, str, list 类型的修改，一旦修改，则 完全覆盖.
    
    from mmengine import Config
    cfg_file = "resnet50_lr0.01.py"
    cfg = Config.fromfile(cfg_file)
    print(cfg)

    # 此外，除却 对字典 的 key 的修改，而且还可以 删除字典对应的 key 
    # 具体方式，则是 在继承后，重新定义新的字典，且 生成 __delete__ = True
    # 随后，就可以将 没有包括的 key 全部删除
    # 这种方式，和没有继承的 区别在什么呢？ 
    # 换句话来说，在这种情况下，对 _base_ 进行了 怎样的继承？
    cfg_file = "resnet50_delete_key.py"
    cfg = Config.fromfile(cfg_file)
    print(cfg.optimizer)

if stage == 5: 
    # 引入 _base_ 配置文件的 变量，并且 修改其中部分内容
    """
    需要明确的是，从 modify_base_var.py 中可以看到，
    这里 现在，已经不用 a = {{_base_.model}} 这样的语法了，而是 直接使用 
    a = _base_.model 的语法。 
    不仅是 为了简洁，而且 这种方法，可以使得后面的 a.model 得以修改

    """
    from mmengine import Config 
    cfg_file = "modify_base_var.py"

    cfg = Config.fromfile(cfg_file)
    print(cfg)

if stage == 6:
    # 检测 预设字段 
    from mmengine import Config 

    cfg_file = "predefined_var.py"
    cfg = Config.fromfile(cfg_file)
    print(cfg)

if stage == 7:
    # custom imports  -- custom_imports.py
    pass 

if stage == 8:
    # Modify cfg from cmd line  --> demo_train.py
    pass 