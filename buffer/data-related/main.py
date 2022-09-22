# data related operation 

stage = 3

if stage == 1: 
    """
    BaseDataElement 的 使用 
    -- 1. 数据元素的创建
    -- 2. new, clone 
    """
    from mmengine.structures.base_data_element import BaseDataElement
    import torch 
    from collections import OrderedDict 

    # 1. Create 
    gt_instances = BaseDataElement()
    print("raw gt_instances: {}".format(gt_instances))

    bboxes = torch.randn([5, 4])
    scores = torch.randn([5,])
    img_id = 0
    H, W = 800, 1333 
    
    # 直接 设置 data参数 和 metainfo参数
    gt_instances = BaseDataElement(
        bboxes=bboxes, scores=scores, 
        metainfo=dict(
            img_id=img_id, img_shape=(H, W)
        ))
    print("gt_instances: {}".format(gt_instances))

    # 2. new, clone 
    # clone 是类似于 copy.deepcopy() 的 深度拷贝，而 
    # new 则是可以看作是 浅度拷贝，且 可以直接进行 数值的覆盖
    gt_instances1 = gt_instances.clone()
    # print("gt_instances1: ", gt_instances1)

    gt_instances2 = gt_instances.new(
        metainfo=dict(img_shape=(800, 1444)), 
        bboxes=torch.randn([3, 4]), 
        scores=torch.randn([3])
    )
    print("gt_instances2", gt_instances2)

elif stage == 2: 
    """
    给出 BaseElement 中，属性的 添加、查询 和 删除 三个常见的操作，

    特别需要注意的是, BaseDataElement 不支持 类似于 字典的索引方式，这主要是考虑到
    不喝 InstanceData 和 PixelData 的索引、切片 相互矛盾
    """
    from mmengine.structures.base_data_element import BaseDataElement 
    import torch 

    gt_instances = BaseDataElement()
    # 默认，直接使用 .attribute 的方式所添加的，都是 data field.
    # 如果需要设置 metainfo，则需要通过 .set_metainfo 的方式
    # 随后，在访问的时候，如果使用的是 迭代器的方式来访问的话，则： 
    # 1. .keys(), .items(), .values() 指的都是 单纯的 data field 的操作; 
    # 2. .all_keys(), .all_items(), .all_values() 指的都是 metainfo + data field 的数据;
    # 3. .meta_keys(), .meta_items(), .meta_values() 指的都是 metainfo field 的数据。
    # 
    # 反之，如果目标仅仅只是某个属性的话，则可以 直接使用 .attribute 来访问，
    # 此时 不需要考虑 所属范围，也就是说，既可以 访问 metainfo 也可以访问 data field 的数据。
    gt_instances.bboxes = torch.randn([3, 4])
    gt_instances.scores = torch.randn([3,])
    gt_instances.set_metainfo(dict(img_shape=(100, 100)))

    print("gt_instances: ", gt_instances)

    assert "img_shape" in gt_instances.all_keys()
    assert "img_shape" not in gt_instances.keys()
    assert "img_shape" in gt_instances.metainfo_keys()
    assert "img_shape" in gt_instances  # call img_shape directly

    assert "bboxes" in gt_instances.all_keys()
    assert "bboxes" in gt_instances.keys()
    assert "bboxes" in gt_instances  # call bboxes directly，这里可以认为是 属性的查询
    
    try: 
        for k, v in gt_instances:
            print("k: ", k, "v: ", v)
    except Exception as e:  # goto this branch 
        print("gt_instances 无法被直接迭代")

elif stage == 3: 
    """
    测试 InstanceData 的使用方式. 

    需要注意的是， TextDetSample 指的就是， 包括了 .gt_instances 和 .pred_instances 两个 
    InstanceData 属性的 类.
    而 TextDetSampleList 指的就是 List[TextDetSample]
    """
    from mmengine.structures import InstanceData
    import torch 
    import numpy as np 

    # 0. InstanceData 的数据校验 
    # 当前的 InstanceData 的属性访问方式，是按照 .attribute 的方式访问的
    img_meta = dict(img_shape=(800, 800, 3), pad_shape=(800, 1216, 3))
    instance_data = InstanceData(metainfo=img_meta)

    instance_data.det_labels = torch.rand([2])
    instance_data.det_boxes = torch.rand([2, 4])
    # print(len(instance_data))

    try: 
        instance_data.scores = torch.rand([3])  # InstanceData 的长度校验
    except Exception as e: 
        print(e)
    
    # 1. 实际上， InstanceData 数据访问，还可以 按照 字典的形式来访问
    # 所以，截止到现在， InstanceData 的数据访问，存在 两种形式： 
    # 1.1 InstanceData.attribute
    # 1.2 InstanceData["attribute"]
    # print(instance_data["det_labels"])
    instance_data["scores"] = torch.rand([2])
    # print(instance_data)

    # 2. InstanceData 的 高级索引属性
    pass 

    # 3. InstanceData 的 拼接
    img_meta = dict(
        img_shape=(300, 800, 3), pdg_shape=(1280, 800, 3), another_name="img_meta")

    instance_data1 = InstanceData(metainfo=img_meta)
    instance_data1.labels = torch.randn(3)
    instance_data1.boxes = torch.randn([3, 4])
    instance_data1.scores = torch.randn(3)

    instance_data2 = InstanceData(metainfo=img_meta)
    instance_data2.labels = torch.randn(1)
    instance_data2.boxes = torch.randn([1, 4])
    instance_data2.scores = torch.randn([1])

    instance_data3 = InstanceData.cat([instance_data1, instance_data2])
    print(instance_data1, "\n", instance_data2, "\n", instance_data3)
