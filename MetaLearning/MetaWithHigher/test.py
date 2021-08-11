
def run_test_epoch(meta_theta, meta_optim, tasksets, args):
    meta_optim.zero_grad()
    meta_theta.train()

    # Meta training set
    X, y = tasksets.test.sample()
    X, y = X.to(args.device), y.to(args.device)
    fast_optim = optim.SGD(meta_theta.parameters(), lr=args.fast_lr, momentum=0.9)
    #Adam(meta_theta.parameters(), lr=args.fast_lr)

    meta_train_indices, meta_test_indices = utils.get_indices(X=X, args=args)
    # for epoch in range(0, args.num_epochs):
    for task in range(args.num_tasks):
        # def train(meta_theta, inner_optimizer, copy_initial_weights, track_higher_grads):
        with higher.innerloop_ctx(meta_theta, fast_optim,
                                  trach_higher_grad=False,
                                  override={'lr': torch.tensor([args.fast_lr],
                                   requires_grad=True).to(args.device)}) as (theta_pi, diff_optim):
            spt_mean_loss = 0
            spt_mean_acc = 0
            for step in range(args.adaptation_steps):
                y_spt = theta_pi(X[meta_train_indices])
                spt_mean_acc += utils.accuracy(y_spt, y[meta_train_indices])
                spt_loss = F.cross_entropy(y_spt, y[meta_train_indices])
                spt_mean_loss += spt_loss.detach().item()
                diff_optim.step(spt_loss)  # After this call. There will be next version of the theta
            #     # https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for

            spt_mean_loss /= args.adaptation_steps
            spt_mean_acc /= args.adaptation_steps
            y_qry = theta_pi(X[meta_test_indices])
            qry_loss = F.cross_entropy(y_qry, y[meta_test_indices])
            qry_acc = utils.accuracy(y_qry, y[meta_test_indices])
            # qry_loss.backward()

    # meta_optim.step()
    return spt_mean_loss, qry_loss.detach().item(), spt_mean_acc, qry_acc
