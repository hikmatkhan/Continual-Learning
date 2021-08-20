import torch
import torch.nn.functional as F
# from MetaLearning.utility import utils
# from . import utils
from torch import nn, optim

import utils
from highers import higher
import mlogger

def run_inner_loop(meta_theta, tasksets, args):
    meta_theta.train()

    meta_train_loss = mlogger.metric.Average()
    meta_train_acc = mlogger.metric.Average()

    for task in range(args.num_tasks):
        # Meta training set
        X, y = tasksets.train.sample()
        X, y = X.to(args.device), y.to(args.device)

        meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
        fast_optim = optim.Adam(meta_theta.parameters(), lr=args.fast_lr)
        with higher.innerloop_ctx(meta_theta, fast_optim,
                                  copy_initial_weights=False,
                                  override={'lr': torch.tensor([args.fast_lr],
                                  requires_grad=True).to(args.device)}
                                  ) as (theta_pi, diff_optim):
            for step in range(args.adaptation_steps):
                mu_y_out, sigma_y_out = theta_pi(X[meta_train_indices])
                labels = nn.functional.one_hot(y[meta_train_indices],
                                   list(theta_pi.children())[-2].out_features)
                s_loss = theta_pi.batch_loss(mu_y_out, sigma_y_out, labels)
                # meta_train_loss.update(s_loss.detach().item())
                diff_optim.step(s_loss)  # After this call. There will be next version of the theta
                # meta_train_acc.update(utils.accuracy(mu_y_out, y[meta_train_indices]))
        #     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for

            mu_y_out, sigma_y_out = theta_pi(X[meta_test_indices])
            labels = nn.functional.one_hot(y[meta_test_indices],
                                           list(theta_pi.children())[-2].out_features)
            q_loss = theta_pi.batch_loss(mu_y_out, sigma_y_out, labels)
            q_loss.backward() # Should accumulate meta-gradients across all tasks.

            meta_train_loss.update(q_loss.detach().item())
            meta_train_acc.update(utils.accuracy(mu_y_out, y[meta_test_indices]))

    return meta_train_acc.value, meta_train_loss.value
        # task_level_train_spt_loss, task_level_train_qry_loss, task_level_train_spt_acc, task_level_train_qry_acc
