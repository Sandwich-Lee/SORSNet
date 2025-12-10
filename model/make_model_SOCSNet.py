import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip import clip
from .cov_matrix import cov_matrix
from model.spdnet_bn.spd import nn as nn_spd

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


def spd_logeuclidean_mean_batch(X_list, weights=None, eps=1e-6):
    """
    计算多个SPD矩阵的Log-Euclidean均值（Fréchet mean）
    X_list: list[Tensor]，每个形状为 (B, n, n)
    weights: list或Tensor，长度与X_list相同，和为1
    """
    N = len(X_list)
    if weights is None:
        weights = [1.0 / N] * N
    else:
        weights = torch.tensor(weights, device=X_list[0].device, dtype=X_list[0].dtype)
        weights = weights / weights.sum()

    # 累积加权log
    log_sum = 0
    for X, w in zip(X_list, weights):
        u, s, _ = torch.linalg.svd(X)
        s = s.log()  # 对数
        logX = u @ torch.diag_embed(s) @ (u.transpose(-1, -2))  # 重新构造对称矩阵
        # eigvals, eigvecs = torch.linalg.eigh(X)
        # eigvals = torch.clamp(eigvals, min=eps)
        # logX = eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(-1, -2)
        log_sum = log_sum + w * logX
    # return log_sum
    # exp回流形
    s, u = torch.linalg.eigh(log_sum)
    expM = u @ torch.diag_embed(torch.exp(s)) @ u.transpose(-1, -2)
    return expM




class SymmetricDropout(nn.Module):
    """
    Symmetric Random Dropout to the SPD matrices
    - keep the Symmetric
    - Can prevent positive definiteness (by introducing small disturbances)
    """
    def __init__(self, drop_rate=0.25, eps=1e-3):
        """
        drop_rate: （0~1）
        eps: To maintain positive definiteness
        """
        super(SymmetricDropout, self).__init__()
        self.drop_rate = drop_rate
        self.eps = eps

    def forward(self, X):
        """
        X: (B, D, D)
        """
        if not self.training or self.drop_rate == 0.0:
            return X
        B, D, _ = X.shape
        # Symmetric mask
        mask = (torch.rand(D, device=X.device) > self.drop_rate).float()  # (D,)
        mask_matrix = torch.ger(mask, mask)  #  (D, D)
        # mask
        X_masked = X * mask_matrix
        # To maintain positive definiteness
        X_masked = X_masked + self.eps * torch.eye(D, device=X.device).unsqueeze(0)
        return X_masked


class Frechet(nn.Module):

    def __init__(self):
        super(Frechet, self).__init__()
        self.logeig = nn_spd.LogEig()
        self.exp = nn_spd.ExpEig()

    def forward(self, X_list, weights=None):
        """
        Log-Euclidean mean (Fréchet mean) of multiple SPD matrices
        X_list: list[Tensor]，with shape (B, n, n)
        weights: list or Tensor，same with X_list, sum of total: 1.0
        """
        N = len(X_list)
        if weights is None:
            weights = [1.0 / N] * N

        log_sum = 0
        for X, w in zip(X_list, weights):
            log_sum = log_sum + self.logeig(X) * w
        expM = self.exp(log_sum)
        return expM


class SPDNet(nn.Module):
    def __init__(self, dim: int, dim1: int, dim2: int, dim3: int, drop_rate, class_num):

        super(__class__, self).__init__()
        """ 
        dim  = origin dimension
        dim1 = dimension after bimap1
        dim2 = dimension after bimap2
        dim3 = dimension after bimap3
        """
        self.sdo = SymmetricDropout(drop_rate=drop_rate)
        self.re = nn_spd.ReEig()

        self.bimap1 = nn_spd.BiMap(1, 1, dim, dim1)
        self.bimap2 = nn_spd.BiMap(1, 1, dim1, dim2)
        self.bimap3 = nn_spd.BiMap(1, 1, dim2, dim3)

        self.logeig = nn_spd.LogEig()

        self.bimap12 = nn_spd.BiMap(1, 1, dim, dim2)
        self.bimap13 = nn_spd.BiMap(1, 1, dim, dim3)
        self.bimap23 = nn_spd.BiMap(1, 1, dim1, dim3)

        self.frechet = Frechet()

    def forward(self, x):
        x = self.sdo(x)
        if len(x.shape) != 4: x = x.unsqueeze(1)

        x12 = self.bimap12(x)
        x12 = self.re(x12)
        x13 = self.bimap13(x)
        x13 = self.re(x13)

        x_spd = self.re(self.bimap1(x))

        x23 = self.bimap23(x_spd)
        x23 = self.re(x23)

        x_spd = self.re(self.bimap2(x_spd))

        x_spd = self.frechet([x_spd, x12])

        x_spd = self.re(self.bimap3(x_spd))

        x_spd = self.frechet([x_spd, x13, x23])

        x_vec = self.logeig(x_spd).view(x_spd.shape[0], -1)  # out_dim^2

        return x_vec

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

        # channels
        dimc = cfg.MODEL.SPD_DIM[0]
        dimc1 = cfg.MODEL.SPD_DIM[1]
        dimc2 = cfg.MODEL.SPD_DIM[2]
        dimc3 = cfg.MODEL.SPD_DIM[3]

        self.spd_out_planes_c = dimc3 ** 2  # 768  # (1 + dimc3) * dimc3 // 2  # 528  #self.num_classes  # 528  #2080 52
        self.bottleneck_spd_c = nn.BatchNorm1d(self.spd_out_planes_c)
        self.bottleneck_spd_c.bias.requires_grad_(False)
        self.classifier_spd_c = nn.Linear(self.spd_out_planes_c, self.num_classes, bias=False)

        self.bottleneck_spd_c.apply(weights_init_kaiming)
        self.classifier_spd_c.apply(weights_init_classifier)

        self.spd_netc = SPDNet(dim=dimc, dim1=dimc1, dim2=dimc2, dim3=dimc3, drop_rate=cfg.MODEL.DR, class_num=num_classes)

    def forward(self, x=None, cam_label=None, view_label=None):

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
        patches = image_features[:,1:]
        patches = patches.view(B, T, patches.shape[1], -1).contiguous()
        patches = patches.view(B, T*128, -1)
        # patches_mean = patches.mean(1)

        img_feature = image_features[:, 0]  # torch.Size([64, 768])
        img_feature = img_feature.view(B, T, -1)  # torch.Size([16, 8, 768])
        img_feature_mean = img_feature.mean(1)  # torch.Size([16, 768])

        feat = self.bottleneck(img_feature_mean)  # torch.Size([16, 768])

        cov_c = cov_matrix(patches.permute(0, 2, 1))  # b, c, T, input_cls_m=img_feature_mean
        video_feature_c_spd = self.spd_netc(cov_c)  # [32, 528]
        spd_c = self.bottleneck_spd_c(video_feature_c_spd)

        if self.training:
            cls_score = self.classifier2(feat)
            spd_c_fl = self.classifier_spd_c(spd_c)

            return [cls_score, spd_c_fl], [img_feature_mean, video_feature_c_spd]
        else:
            if self.neck_feat == 'after':
                print("Test with feature after BN")
                return torch.cat([spd_c], dim=1)
            else:
                return torch.cat([video_feature_c_spd], dim=1)

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


def make_model(cfg, num_class, camera_num, view_num):  # 0
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model
