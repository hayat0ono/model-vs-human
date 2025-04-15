#!/usr/bin/env python3
import torch

from ..registry import register_model
from ..wrappers.pytorch import PytorchModel, PyContrastPytorchModel, ClipPytorchModel, \
    ViTPytorchModel, EfficientNetPytorchModel, SwagPytorchModel

_PYTORCH_IMAGE_MODELS = "rwightman/pytorch-image-models"

_EFFICIENTNET_MODELS = "rwightman/gen-efficientnet-pytorch"


def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN(model_name, *args):
    from .shapenet import texture_shape_models as tsm
    model = tsm.load_model(model_name)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet9(model_name, *args):
    from .bagnets.pytorchnet import bagnet9
    model = bagnet9(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet17(model_name, *args):
    from .bagnets.pytorchnet import bagnet17
    model = bagnet17(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet33(model_name, *args):
    from .bagnets.pytorchnet import bagnet33
    model = bagnet33(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1_supervised_baseline(model_name, *args):
    from .simclr import simclr_resnet50x1_supervised_baseline
    model = simclr_resnet50x1_supervised_baseline(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4_supervised_baseline(model_name, *args):
    from .simclr import simclr_resnet50x4_supervised_baseline
    model = simclr_resnet50x4_supervised_baseline(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1(model_name, *args):
    from .simclr import simclr_resnet50x1
    model = simclr_resnet50x1(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x2(model_name, *args):
    from .simclr import simclr_resnet50x2
    model = simclr_resnet50x2(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4(model_name, *args):
    from .simclr import simclr_resnet50x4
    model = simclr_resnet50x4(pretrained=True,
                              use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def InsDis(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import InsDis
    model, classifier = InsDis(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCo(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import MoCo
    model, classifier = MoCo(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCoV2(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import MoCoV2
    model, classifier = MoCoV2(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def PIRL(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import PIRL
    model, classifier = PIRL(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def InfoMin(model_name, *args):
    from .pycontrast.pycontrast_resnet50 import InfoMin
    model, classifier = InfoMin(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0
    model = resnet50_l2_eps0()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_01(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_01
    model = resnet50_l2_eps0_01()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_03(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_03
    model = resnet50_l2_eps0_03()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_05(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_05
    model = resnet50_l2_eps0_05()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_1(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_1
    model = resnet50_l2_eps0_1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_25(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_25
    model = resnet50_l2_eps0_25()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_5(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_5
    model = resnet50_l2_eps0_5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps1(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps1
    model = resnet50_l2_eps1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps3(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps3
    model = resnet50_l2_eps3()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps5(model_name, *args):
    from .adversarially_robust.robust_models import resnet50_l2_eps5
    model = resnet50_l2_eps5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_es(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0_noisy_student(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "tf_efficientnet_b0_ns",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_l2_noisy_student_475(model_name, *args):
    model = torch.hub.load(_EFFICIENTNET_MODELS,
                           "tf_efficientnet_l2_ns_475",
                           pretrained=True)
    return EfficientNetPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B16_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('B_16_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B32_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('B_32_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L16_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('L_16_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L32_IN21K(model_name, *args):
    from pytorch_pretrained_vit import ViT
    model = ViT('L_32_imagenet1k', pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def vit_small_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_base_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_large_patch16_224(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def cspresnet50(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspresnext50(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspdarknet53(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def darknet53(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn92(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn98(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn131(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn107(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small_v2(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w30(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w40(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w44(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w48(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w64(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls84(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60b(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           model_name,
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def clip(model_name, *args):
    import clip
    model, _ = clip.load("ViT-B/32")
    return ClipPytorchModel(model, model_name, *args)


@register_model("pytorch")
def clipRN50(model_name, *args):
    import clip
    model, _ = clip.load("RN50")
    return ClipPytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_swsl(model_name, *args):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnet50_swsl')
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def ResNeXt101_32x16d_swsl(model_name, *args):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext101_32x16d_swsl')
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x1(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_50x1_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x3(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_50x3_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x1(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_101x1_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x3(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_101x3_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x2(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_152x2_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x4(model_name, *args):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS,
                           "resnetv2_152x4_bitm",
                           pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_hard_labels(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
                                                    "/ResNet50_clip_hard_labels.pth",map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_soft_labels(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
                                                    "/ResNet50_clip_soft_labels.pth", map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def swag_regnety_16gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_16gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_32gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_32gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_128gf_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="regnety_128gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_b16_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_l16_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_l16_in1k")
    return SwagPytorchModel(model, model_name, input_size=512, *args)


@register_model("pytorch")
def swag_vit_h14_in1k(model_name, *args):
    model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
    return SwagPytorchModel(model, model_name, input_size=518, *args)


@register_model("pytorch")
def resnet50_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr2(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr2', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr3(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr3', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr4(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr4', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr5(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr5', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr6(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr6', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr7(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr7', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr8(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr8', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr9(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr9', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr10(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr10', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr11(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr11', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr12(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr12', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr13(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr13', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr14(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr14', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr15(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr15', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr16(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr16', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr17(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr17', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr18(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr18', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr19(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr19', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr20(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr20', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr21(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr21', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr22(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr22', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr23(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr23', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr24(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr24', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr25(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr25', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr26(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr26', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr27(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr27', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr01_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.1-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr02_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.2-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr03_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.3-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr04_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.4-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr05_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.5-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr09_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.9-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr10_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr1.0-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr06_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.6-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr07_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.7-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_random_pr08_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-random-pr0.8-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_style_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-style_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr28_content_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr28-content_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_itr29(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('resnet50-itr29', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr2(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr2', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr3(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr3', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr4(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr4', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr5(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr5', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr6(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr6', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr7(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr7', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr8(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr8', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr9(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr9', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr10(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr10', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr11(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr11', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr12(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr12', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr13(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr13', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr14(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr14', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr15(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr15', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr16(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr16', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr17(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr17', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr18(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr18', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr19(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr19', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr20(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr20', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr21(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr21', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr22(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr22', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr23(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr23', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr24(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr24', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr25(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr25', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr26(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr26', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr27(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr27', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr28(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr28', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr29(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr29', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr30(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr30', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr31(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr31', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr32(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr32', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr33(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr33', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr34(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr34', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr35(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr35', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr36(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr36', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr37(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr37', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr38(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr38', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr39(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr39', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr01_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.1-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr02_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.2-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr03_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.3-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr04_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.4-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr05_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.5-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr09_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.9-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr10_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr1.0-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr06_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.6-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr07_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.7-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_random_pr08_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-random-pr0.8-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_content_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-content_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr40_style_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr40-style_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_itr41(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('alexnet-itr41', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr2(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr2', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr3(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr3', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr4(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr4', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr5(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr5', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr6(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr6', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr7(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr7', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr8(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr8', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr9(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr9', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr10(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr10', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr11(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr11', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr12(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr12', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr13(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr13', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr14(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr14', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr15(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr15', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr16(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr16', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr17(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr17', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr18(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr18', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr19(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr19', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr20(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr20', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr21(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr21', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr22(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr22', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr23(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr23', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr24(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr24', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr01_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.1-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr02_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.2-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr03_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.3-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr04_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.4-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr05_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.5-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr09_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.9-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr10_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr1.0-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr06_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.6-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr07_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.7-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_random_pr08_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-random-pr0.8-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_style_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-style_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr25_content_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr25-content_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg19_itr26(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('vgg19-itr26', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr2(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr2', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr3(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr3', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr4(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr4', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr5(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr5', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr6(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr6', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr7(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr7', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr8(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr8', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr9(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr9', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr10(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr10', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr11(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr11', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr12(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr12', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr13(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr13', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr14(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr14', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr15(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr15', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr16(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr16', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr17(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr17', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr18(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr18', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr19(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr19', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr20(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr20', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr21(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr21', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr01_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.1-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr02_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.2-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr03_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.3-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr04_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.4-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr05_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.5-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr09_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.9-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr10_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr1.0-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr06_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.6-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr07_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.7-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial1_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial1_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial2_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial2_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial3_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial3_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial4_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial4_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial5_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_random_pr08_trial5_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-random-pr0.8-trial5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_average_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_average-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_max_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_max-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_min_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_min-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_style_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-style_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr01_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.1-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr01_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.1-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr02_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.2-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr02_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.2-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr03_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.3-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr03_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.3-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr04_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.4-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr04_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.4-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr05_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.5-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr05_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.5-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr09_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.9-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr09_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.9-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr10_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr1.0-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr10_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr1.0-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr06_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.6-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr06_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.6-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr07_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.7-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr07_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.7-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr08_itr0(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.8-itr0', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr22_content_score_entropy_pr08_itr1(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr22-content_score_entropy-pr0.8-itr1', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cornet_s_itr23(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('cornet_s-itr23', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)
