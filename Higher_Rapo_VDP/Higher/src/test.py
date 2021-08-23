import mlogger
import torch.nn.functional as F
from torch import optim
import utils
import higher


def run_test_loop(meta_theta, tasksets, args):
    meta_test_loss = mlogger.metric.Average()
    meta_test_acc = mlogger.metric.Average()
    meta_theta.eval()
    for t in range(args.num_tasks):
        X, y = tasksets.test.sample()
        X, y = X.to(args.device), y.to(args.device)
        meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
        optim_fast = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
        with higher.innerloop_ctx(model=meta_theta, opt=optim_fast, track_higher_grads=False
                                  ) as (fmodel, diff_opp):
            for a in range(args.adaptation_steps):
                y_prd = fmodel(X[meta_train_indices])
                #                     a_meta_train_spt_acc.update(accuracy(y_prd, y[meta_train_indices]))
                meta_train_spt_loss = F.cross_entropy(y_prd, y[meta_train_indices])
                #                     a_meta_train_spt_loss.update(meta_train_spt_loss.detach().cpu().item())
                diff_opp.step(meta_train_spt_loss)
            #                 meta_train_spt_loss.update(a_meta_train_spt_loss.value)
            #                 meta_train_spt_acc.update(a_meta_train_spt_acc.value)

            y_prd = fmodel(X[meta_test_indices])
            meta_test_acc.update(utils.accuracy(y_prd, y[meta_test_indices]))
            meta_train_qry_loss = F.cross_entropy(y_prd, y[meta_test_indices])
            meta_test_loss.update(meta_train_qry_loss.detach().cpu().item())
            # meta_train_qry_loss.backward()
    return round(meta_test_acc.value, 3), round(meta_test_loss.value, 3)


def run_val_loop(meta_theta, tasksets, args):
    meta_val_loss = mlogger.metric.Average()
    meta_val_acc = mlogger.metric.Average()
    meta_theta.eval()
    for t in range(args.num_tasks):
        X, y = tasksets.validation.sample()
        X, y = X.to(args.device), y.to(args.device)
        meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
        optim_fast = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
        with higher.innerloop_ctx(model=meta_theta, opt=optim_fast, track_higher_grads=False
                                  ) as (fmodel, diff_opp):
            for a in range(args.adaptation_steps):
                y_prd = fmodel(X[meta_train_indices])
                #                     a_meta_train_spt_acc.update(accuracy(y_prd, y[meta_train_indices]))
                meta_train_spt_loss = F.cross_entropy(y_prd, y[meta_train_indices])
                #                     a_meta_train_spt_loss.update(meta_train_spt_loss.detach().cpu().item())
                diff_opp.step(meta_train_spt_loss)
            #                 meta_train_spt_loss.update(a_meta_train_spt_loss.value)
            #                 meta_train_spt_acc.update(a_meta_train_spt_acc.value)

            y_prd = fmodel(X[meta_test_indices])
            meta_val_acc.update(utils.accuracy(y_prd, y[meta_test_indices]))
            meta_train_qry_loss = F.cross_entropy(y_prd, y[meta_test_indices])
            meta_val_loss.update(meta_train_qry_loss.detach().cpu().item())
            # meta_train_qry_loss.backward()
    return round(meta_val_acc.value, 3), round(meta_val_loss.value, 3)
