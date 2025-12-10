from bisect import bisect_right
import torch

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,  # steps
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


# new
class WarmupMultiStepStiefelLR:
    """
    针对自定义流形优化器（例如 StiefelOptim）实现的带 warmup 和 multi-step 衰减的调度器，
    该调度器直接修改优化器对象的 lr 属性。
    """

    def __init__(
            self,
            optimizer,  # 自定义优化器实例，比如 StiefelOptim
            milestones,  # 衰减 step 列表，如 [1000, 2000]
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of increasing integers. Got {}".format(milestones))
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method))

        self.optimizer = optimizer  # 传入自定义优化器实例
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.last_epoch = last_epoch

        # 保存初始 lr
        self.base_lr = optimizer.optimizer.param_groups[0]['lr']  # 获取内部优化器的学习率
        self._initial_step()

    def _initial_step(self):
        self.step()

    def step(self):
        self.last_epoch += 1

        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        else:
            warmup_factor = 1.0

        decay = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        new_lr = self.base_lr * warmup_factor * decay

        # 更新自定义优化器的 lr
        # 更新 StiefelMetaOptimizer 的学习率
        for group in self.optimizer.optimizer.param_groups:
            group['lr'] = new_lr

    def get_lr(self):
        return self.optimizer.optimizer.param_groups[0]['lr']


class WarmupMultiStepMixLR:

    def __init__(self, mix_optimizer, milestones, gamma=0.1,
                 warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear"):
        self.mix_optimizer = mix_optimizer

        self.regular_scheduler = WarmupMultiStepLR(
            self.mix_optimizer.optimizer,
            milestones=milestones,
            gamma=gamma,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_iters,
            warmup_method=warmup_method,
            last_epoch=-1
        )
        # if len(self.mix_optimizer.params_stiefel):
        #     # for stefiel manifold
        #     self.stiefel_scheduler = WarmupMultiStepStiefelLR(
        #         self.mix_optimizer.stiefel_optim,
        #         milestones=milestones,
        #         gamma=gamma,
        #         warmup_factor=warmup_factor,
        #         warmup_iters=warmup_iters,
        #         warmup_method=warmup_method,
        #         last_epoch=-1
        #     )

    def step(self):
        # for regular optimizer to update the lr of param_group
        self.regular_scheduler.step()
        # for Stiefel optimizer
        # if len(self.mix_optimizer.params_stiefel):
        #     # update the lr of Stiefel manifold
        #     self.stiefel_scheduler.step()
    def get_lr(self):
        # return both lr info
        regular_lr = (self.mix_optimizer.optimizer.param_groups[0]['lr']
                      if self.mix_optimizer.optimizer.param_groups else None)
        # if len(self.mix_optimizer.params_stiefel):
        #     stiefel_lr = self.stiefel_scheduler.get_lr()
        #     return [regular_lr, stiefel_lr]
        return [regular_lr, 0.]
# ***