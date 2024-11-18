from torch import optim


class options():
    def __init__(self, cfg, model):
        self.optimizer = cfg["params"]["optimizer"]

        self.adagrad = optim.Adagrad(
            params=model.parameters(),
            lr=cfg["params"]["lr"],
            lr_decay=cfg["params"]["lr_decay"],
            weight_decay=cfg["params"]["weight_decay"],
            initial_accumulator_value=cfg["params"][
                "initial_accumulator_value"
            ],
            eps=cfg["params"]["eps"]
        )
        self.adam = optim.Adam(
            params=model.parameters(),
            lr=cfg["params"]["lr"],
            betas=tuple(cfg["params"]["betas"]),
            eps=cfg["params"]["eps"],
            weight_decay=cfg["params"]["weight_decay"],
            amsgrad=cfg["params"]["amsgrad"],
            foreach=cfg["params"]["foreach"],
            maximize=cfg["params"]["maximize"],
            capturable=cfg["params"]["capturable"],
            differentiable=cfg["params"]["differentiable"],
            fused=cfg["params"]["fused"]
        )
        self.adamw = optim.AdamW(
            params=model.parameters(),
            lr=cfg["params"]["lr"],
            betas=tuple(cfg["params"]["betas"]),
            eps=cfg["params"]["eps"],
            weight_decay=cfg["params"]["weight_decay"],
            amsgrad=cfg["params"]["amsgrad"],
            maximize=cfg["params"]["maximize"],
            foreach=cfg["params"]["foreach"],
            capturable=cfg["params"]["capturable"],
            differentiable=cfg["params"]["differentiable"],
            fused=cfg["params"]["fused"]
        )
        self.asgd = optim.ASGD(
            params=model.parameters(),
            lr=cfg["params"]["lr"],
            lambd=cfg["params"]["lambd"],
            alpha=cfg["params"]["alpha"],
            t0=cfg["params"]["t0"],
            weight_decay=cfg["params"]["weight_decay"]
        )
        self.radam = optim.RAdam(
            params=model.parameters(),
            lr=cfg["params"]["lr"],
            betas=tuple(cfg["params"]["betas"]),
            eps=cfg["params"]["eps"],
            weight_decay=cfg["params"]["weight_decay"],
            decoupled_weight_decay=cfg["params"]["decoupled_weight_decay"]
        )
        self.sgd = optim.SGD(
            params=model.parameters(),
            lr=cfg["params"]["lr"],
            momentum=cfg["params"]["momentum"],
            dampening=cfg["params"]["dampening"],
            weight_decay=cfg["params"]["weight_decay"],
            nesterov=cfg["params"]["nesterov"]
        )

    def getter(self):
        opt_dict = {
            "adagrad": self.adagrad,
            "adam": self.adam,
            "adamw": self.adamw,
            "asgd": self.asgd,
            "radam": self.radam,
            "sgd": self.sgd
        }
        return opt_dict[self.optimizer]
