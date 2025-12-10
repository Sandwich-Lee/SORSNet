import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip import clip

_tokenizer = _Tokenizer()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = clip._download(url)  #
    # model_path = '/dataset_cc/Pretrain-models/ViT-B-16.pt'  # downloaded
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class build_transformer(nn.Module):  # 1
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME

        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE  # 1

        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(512)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

    def forward(self, x=None, cam_label=None, view_label=None):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(1)  # [B, C, H, W] → [B, 1, C, H, W]
        elif len(x.shape) == 5:
            B, T, C, H, W = x.shape
        B, T, C, H, W = x.shape  # B=6, T=8, C=3, H=256,W=128
        x = x.view(-1, C, H, W)  # torch.Size([64*4, 3, 256, 128])

        if cam_label != None and view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
        elif cam_label != None:  # 1
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
            cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
        elif view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
            cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
        else:
            cv_embed = None
        # cv_embed = cv_embed.repeat((1, B)).view(B, -1)  # torch.Size([64, 768])
        # cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
        # torch.Size([16*8, 129, 768])  torch.Size([16*8 129, 512])
        image_features = self.image_encoder(x, cv_embed)
        img_feature = image_features[:, 0]  # torch.Size([64, 768])
        img_feature = img_feature.view(B, T, -1)  # torch.Size([16, 8, 768])
        img_feature_mean = img_feature.mean(1)  # torch.Size([16, 768])



        feat = self.bottleneck(img_feature_mean)  # torch.Size([16, 768])

        if self.training:

            cls_score = self.classifier2(feat)

            return [cls_score], [img_feature_mean]
        else:
            if self.neck_feat == 'after':
                print("Test with feature after BN")

                return torch.cat([feat], dim=1)
            else:
                return torch.cat([img_feature_mean], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model_baseline(cfg, num_class, camera_num, view_num):  # 0
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model
