from learn2learn.algorithms.base_learner import BaseLearner


class MetaConvNet(BaseLearner):
    def __init__(self, module=None):
        super().__init__(module)
        self.module = module


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self):
        return None

    def clone(self):
        return None