from torch import nn
from torch import optim
import copy


# MAML Agent, Which will implement the MAML Algorithm
class Maml(nn.Module):

    def __init__(self, model, args):
        super(Maml, self).__init__()
        # self.module = mlp.MLP100().to(args.device)
        # torchsummary.summary(
        #     self.module, input_size=(1, 32 * 32))
        self.module = model
        self.optm = optim.Adam(self.module.parameters(), lr=args.fast_lr)
        self.criterion = nn.CrossEntropyLoss()

    # def init_with_meta_theta(self, meta_theta):
    #     self.module.load_state_dict(meta_theta)
    #     print("Init with meta parameters")

    # def get_meta_theta(self):
    #     return self.module.state_dict()

    def forward(self, x):
        return self.module(x)

    def adapt(self, y_true, y_prd):
        self.optm.zero_grad()
        loss = self.criterion(y_prd, y_true)
        loss.backward()
        self.optm.step()
        return loss

    # def fast_step(self, X, y):
    #     y_prd = self.module(X)
    #     loss = self.criterion(y_prd, y)
    #     self.optm.zero_grad()
    #     loss.backward()
    #     self.optm.step()
    #     return loss

    def clone(self):
        # print("clone parameters")
        return copy.deepcopy(self.module)

# if __name__ == '__main__':
#     maml = MAML()
#     maml.adapt()
#     maml.clone()
#     print(maml)
