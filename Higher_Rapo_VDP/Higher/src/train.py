import higher
import mlogger
from torch import optim
import utils
import torch.nn.functional as F


def run_inner_loop(meta_theta, tasksets, args):
    meta_train_loss = mlogger.metric.Average()
    meta_train_acc = mlogger.metric.Average()
    meta_theta.train()
    for t in range(args.num_tasks):
        X, y = tasksets.train.sample()

        X, y = X.to(args.device), y.to(args.device)
        #             print(X.size())
        meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
        optim_fast = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
        with higher.innerloop_ctx(model=meta_theta, opt=optim_fast, copy_initial_weights=False
                                  ) as (fmodel, diff_opp):

            #                 print(fmodel.conv1.weight[0].grad)
            for a in range(args.adaptation_steps):
                y_prd = fmodel(X[meta_train_indices])
                #                     a_meta_train_spt_acc.update(accuracy(y_prd, y[meta_train_indices]))
                meta_train_spt_loss = F.cross_entropy(y_prd, y[meta_train_indices])
                #                     a_meta_train_spt_loss.update(meta_train_spt_loss.detach().cpu().item())
                diff_opp.step(meta_train_spt_loss)
            #                 meta_train_spt_loss.update(a_meta_train_spt_loss.value)
            #                 meta_train_spt_acc.update(a_meta_train_spt_acc.value)

            y_prd = fmodel(X[meta_test_indices])
            meta_train_acc.update(utils.accuracy(y_prd, y[meta_test_indices]))
            meta_train_qry_loss = F.cross_entropy(y_prd, y[meta_test_indices])
            meta_train_loss.update(meta_train_qry_loss.detach().cpu().item())
            meta_train_qry_loss.backward()
    return round(meta_train_acc.value, 3), round(meta_train_loss.value, 3)
