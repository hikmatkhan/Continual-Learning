import torch.nn.functional as F
# from MetaLearning.utility import utils
# from . import utils
from torch import nn

import utils
from highers import higher


def run_inner_loop(meta_theta, fast_optim, tasksets, args):
    meta_theta.train()

    task_level_train_spt_loss = []
    task_level_train_qry_loss = []
    task_level_train_spt_acc = []
    task_level_train_qry_acc = []

    for task in range(args.num_tasks):
    # for (X, y) in tasksets:
        # Meta training set
        X, y = tasksets.train.sample()

        X, y = X.to(args.device), y.to(args.device)
        # print("X:", X.size())
        # print("X:", X.view(len(X), -1).size())
        # continue
        # fast_optim = optim.SGD(meta_theta.parameters(), lr=args.fast_lr, momentum=0.9)
        # Adam(meta_theta.parameters(), lr=args.fast_lr)

        meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
        # for epoch in range(0, args.num_epochs):


        # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
        with higher.innerloop_ctx(meta_theta, fast_optim,
                                  copy_initial_weights=False,
                                  # override={'lr': torch.tensor([args.fast_lr],
                                  # requires_grad=True).to(args.device)}
                                  ) as (theta_pi, diff_optim):
            spt_mean_loss = 0
            spt_mean_acc = 0
            for step in range(args.adaptation_steps):
                # y_spt = theta_pi(X[meta_train_indices])
                # print("Input Dim", X[meta_train_indices].size())
                mu_y_out, sigma_y_out = theta_pi(X[meta_train_indices])

                labels = nn.functional.one_hot(y[meta_train_indices],
                                   list(theta_pi.children())[-2].out_features)
                # print("Label:", labels)
                s_loss = theta_pi.batch_loss(mu_y_out, sigma_y_out, labels)
                # print("mu_y_out:", mu_y_out.size())
                # print("sigma_y_out:", sigma_y_out.size())

                # _, pred = mu_y_out.max(1, keepdim=True)


                # spt_mean_acc += utils.accuracy(mu_y_out, y[meta_train_indices])
                # total_acc += pred.eq(targets.view_as(pred)).sum().item()

                # spt_loss = F.cross_entropy(mu_y_out, y[meta_train_indices])
                # loss.detach()
                spt_mean_loss += s_loss.detach().item()
                diff_optim.step(s_loss)  # After this call. There will be next version of the theta

                spt_mean_acc += utils.accuracy(mu_y_out, y[meta_train_indices])
                # total_acc += pred.eq(targets.view_as(pred)).sum().item()
        #     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for

            spt_mean_loss /= args.adaptation_steps
            spt_mean_acc /= args.adaptation_steps
            # y_qry = theta_pi(X[meta_test_indices])
            mu_y_out, sigma_y_out = theta_pi(X[meta_test_indices])
            # qry_loss = F.cross_entropy(mu_y_out, y[meta_test_indices])
            labels = nn.functional.one_hot(y[meta_train_indices],
                                           list(theta_pi.children())[-2].out_features)
            q_loss = theta_pi.batch_loss(mu_y_out, sigma_y_out, labels)

            qry_acc = utils.accuracy(mu_y_out, y[meta_test_indices])
            # qry_loss.backward()
            q_loss.backward()

            task_level_train_spt_loss.append(spt_mean_loss)
            # task_level_train_qry_loss.append(qry_loss.detach().item())
            task_level_train_qry_loss.append(q_loss.detach().item())
            task_level_train_spt_acc.append(spt_mean_acc)
            task_level_train_qry_acc.append(qry_acc)

    return task_level_train_spt_loss, task_level_train_qry_loss, task_level_train_spt_acc, task_level_train_qry_acc
