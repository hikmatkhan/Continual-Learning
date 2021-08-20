import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import SGD, Adam

import utils
from highers import higher
import mlogger


def run_val_loop(meta_theta, tasksets, args, eval_on_testset):
    meta_theta.eval()
    meta_test_loss = mlogger.metric.Average()
    meta_test_acc = mlogger.metric.Average()
    for task in range(args.num_tasks):

        if eval_on_testset:
            # Meta Test set (Adaptation)
            X, y = tasksets.test.sample()
        else:
            # Meta Evaluation set
            X, y = tasksets.validation.sample()
        X, y = X.to(args.device), y.to(args.device)
        # fast_optim = optim.SGD(meta_theta.parameters(), lr=args.fast_lr, momentum=0.9)
        fast_optim = Adam(meta_theta.parameters(), lr=args.fast_lr)
        meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
        with higher.innerloop_ctx(meta_theta, fast_optim,
                                  track_higher_grads=False,
                                  override={'lr': torch.tensor([args.fast_lr],
                                  requires_grad=True).to(args.device)}
                                  ) as (theta_pi, diff_optim):
            for step in range(args.adaptation_steps):
                mu_y_out, sigma_y_out = theta_pi(X[meta_train_indices])
                labels = nn.functional.one_hot(y[meta_train_indices],
                                               list(theta_pi.children())[-2].out_features)
                s_loss = theta_pi.batch_loss(mu_y_out, sigma_y_out, labels)
                diff_optim.step(s_loss)  # After this call. There will be next version of the theta
            #     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for

            mu_y_out, sigma_y_out = theta_pi(X[meta_test_indices])
            labels = nn.functional.one_hot(y[meta_test_indices],
                                           list(theta_pi.children())[-2].out_features)
            q_loss = theta_pi.batch_loss(mu_y_out, sigma_y_out, labels)
            qry_acc = utils.accuracy(mu_y_out, y[meta_test_indices])
            meta_test_loss.update(q_loss.detach().item())
            meta_test_acc.update(qry_acc)

    return meta_test_acc.value, meta_test_loss.value
    # task_level_spt_loss, task_level_qry_loss, task_level_spt_acc, task_level_qry_acc
