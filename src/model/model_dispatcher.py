import timm


from .my_model import MyModel_v1

# 2 output for training both two labels
from .my_model2out import MyModel_v1_2

# only distortion classification
from .my_model_distonly import MyModel_v1_0, MyModel_v1_4#(for only quality)

def create_model(cfg):
    if cfg.model.model_name == "mymodel_v1":
        basic_model = timm.create_model(
            cfg.model.basic_model_name,
            img_size=cfg.model.vit_param.img_size,
            pretrained=cfg.model.basic_model_pretrained,
            num_classes=cfg.model.vit_param.num_classes,
        )

        net_arch = MyModel_v1(cfg=cfg, basic_state_dict=basic_model.state_dict())
        net_arch.freeze()
    elif cfg.model.model_name == "mymodel_v1_2":
        basic_model = timm.create_model(
            cfg.model.basic_model_name,
            img_size=cfg.model.vit_param.img_size,
            pretrained=cfg.model.basic_model_pretrained,
            num_classes=cfg.model.vit_param.num_classes,
        )

        net_arch = MyModel_v1_2(cfg=cfg, basic_state_dict=basic_model.state_dict())
        net_arch.freeze()
    
    elif cfg.model.model_name == "mymodel_v1_0":
        basic_model = timm.create_model(
            cfg.model.basic_model_name,
            img_size=cfg.model.vit_param.img_size,
            pretrained=cfg.model.basic_model_pretrained,
            num_classes=cfg.model.vit_param.num_classes,
        )

        net_arch = MyModel_v1_0(cfg=cfg, basic_state_dict=basic_model.state_dict())
        net_arch.freeze()

    elif cfg.model.model_name == "mymodel_v1_4":
        basic_model = timm.create_model(
            cfg.model.basic_model_name,
            img_size=cfg.model.vit_param.img_size,
            pretrained=cfg.model.basic_model_pretrained,
            num_classes=cfg.model.vit_param.num_classes,
        )

        net_arch = MyModel_v1_4(cfg=cfg, basic_state_dict=basic_model.state_dict())
        net_arch.freeze()

    else:
        raise Exception("%s model not supported" % cfg.model.model_name)

    return net_arch
