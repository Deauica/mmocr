_base_ = ["resnet50.py"]
# a = {{_base_.model}}
# a["type"] = "MobileNet"
a = _base_.model
a.type = "MobileNet"

"""
a = {{_base_.model}}
a["type"] = "MobileNet" --> Error 
"""

"""
a = {{_base_.model}}
a.type = "MobileNet --> Error
"""

"""
a = _base_.model 
a["type] = "MobileNet --> Error
"""

"""
a = _base_.model
a.type = "MobileNet" --> Correct
"""