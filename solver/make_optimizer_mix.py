import torch
import geoopt
from model.spdnet_bn.spd import optimizer

class MixOptimizer():
    def __init__(self,cfg, model, center_criterion):
        self.params = []
        self.params_stiefel = []
        self.keys = []
        self.lr = cfg.SOLVER.BASE_LR
        self.lr_st = cfg.SOLVER.BASE_LR_STIEFEL
        self.weight_decay = cfg.SOLVER.WEIGHT_DECAY

        for key, value in model.named_parameters():
            if not value.requires_grad:
                    continue
            # if value.__class__.__name__=='StiefelParameter':
            if value.__class__.__name__ == 'ManifoldParameter':
                if "bias" in key:
                    self.lr_st = cfg.SOLVER.BASE_LR_STIEFEL * cfg.SOLVER.BIAS_LR_FACTOR
                    self.weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                self.params_stiefel += [{"params":value, "lr": self.lr_st, "weight_decay": self.weight_decay}]
                self.lr_st = cfg.SOLVER.BASE_LR_STIEFEL
                self.weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                continue
            else:
                if "bias" in key:
                    self.lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    self.weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                if cfg.SOLVER.LARGE_FC_LR:
                    if "classifier_proj_temp" in key or "arcface" in key:
                        self.lr = self.lr * 10
                        print('Using 10 times learning rate for fc ')
                self.params += [{"params": [value], "lr": self.lr, "weight_decay": self.weight_decay}]
                self.lr = cfg.SOLVER.BASE_LR
                self.weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            self.keys += [key]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            self.optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(self.params, momentum=cfg.SOLVER.MOMENTUM)
        elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        else:
            self.optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(self.params)
            if len(self.params_stiefel):
                # self.stiefel_optim = StiefelMetaOptimizer(
                #     getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(self.params_stiefel))  # , weight_decay=self.weight_decay
                self.stiefel_optim = geoopt.optim.RiemannianAdam(self.params_stiefel, lr=self.lr_st)
        # self.optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    def step(self):
        self.optimizer.step()
        if len(self.params_stiefel):
            self.stiefel_optim.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
        if len(self.params_stiefel):
            self.stiefel_optim.zero_grad()