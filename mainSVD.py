from ast import Pass
from enum import Flag
from pickle import TRUE
from unittest.main import MODULE_EXAMPLES
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cvx

import lunzi as lz
from lunzi.typing import *
from opt import GroupRMSprop

import cuda.SVDLinear as SVDLinear
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
X = torch.rand(100,100).to(device).requires_grad_(False)

class FLAGS(lz.BaseFLAGS):
    problem = ''
    gt_path = ''
    obs_path = ''

    depth = 1
    n_train_samples = 0
    n_iters = 1000000
    n_dev_iters = max(1, n_iters // 10000)
    init_scale = 0.001  # average magnitude of entries
    shape = [0, 0]
    n_singulars_save = 0

    optimizer = 'GroupRMSprop'
    initialization = 'gaussian' #'identity'  `orthogonal` or `gaussian` or `gaussian`
    lr = 0.01
    regular_lr = 0.01
    angle_lr = 0.01
    train_thres = 1.e-3
    momentum= 0.9
    hidden_sizes = []

    @classmethod
    def finalize(cls):
        assert cls.problem
        cls.add('hidden_sizes', [cls.shape[0]] + [cls.shape[1]] * cls.depth, overwrite_false=True)


def get_e2e(model:SVDLinear):
    return model.forward(X)


@FLAGS.inject
def init_model(model, *, hidden_sizes, initialization, init_scale, _log):
    depth = len(hidden_sizes) - 1

    if initialization == 'orthogonal':
        pass

    elif initialization == 'identity':
        pass

    elif initialization == 'gaussian':
        # sigmas are manchenko pastur by default
        #torch.nn.init.normal_(model.sigma, std=0.0001)

        e2e = get_e2e(model).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = FLAGS.init_scale * np.sqrt(hidden_sizes[0])

        print(e2e)
        print("\n",e2e_fro)
        print(desired_fro)
        print(e2e_fro / desired_fro)
        _log.info(f"[check] e2e fro norm: {e2e_fro:.6e}, desired = {desired_fro:.6e}")
        #assert 0.8 <= e2e_fro / desired_fro <= 1.2

    elif initialization == 'uniform':
        pass

    else:
        assert 0


class BaseProblem:
    def get_d_e2e(self, e2e):
        pass

    def get_train_loss(self, e2e):
        pass

    def get_test_loss(self, e2e):
        pass

    def get_cvx_opt_constraints(self, x) -> list:
        pass


@FLAGS.inject
def cvx_opt(prob: BaseProblem, *, shape, _log: Logger, _writer: SummaryWriter, _fs: FileStorage):
    x = cvx.Variable(shape=shape)

    objective = cvx.Minimize(cvx.norm(x, 'nuc'))
    constraints = prob.get_cvx_opt_constraints(x)

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.SCS, verbose=True, use_indirect=False)
    e2e = torch.from_numpy(x.value).float()

    train_loss = prob.get_train_loss(e2e)
    test_loss = prob.get_test_loss(e2e)

    nuc_norm = e2e.norm('nuc')
    _log.info(f"train loss = {train_loss.item():.3e}, "
              f"test error = {test_loss.item():.3e}, "
              f"nuc_norm = {nuc_norm.item():.3f}")
    _writer.add_scalar('loss/train', train_loss.item())
    _writer.add_scalar('loss/test', test_loss.item())
    _writer.add_scalar('nuc_norm', nuc_norm.item())

    torch.save(e2e, _fs.resolve('$LOGDIR/nuclear.npy'))


class MatrixCompletion(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, gt_path, obs_path):
        self.w_gt = torch.load(gt_path, map_location=device)
        (self.us, self.vs), self.ys_ = torch.load(obs_path, map_location=device)

    def get_train_loss(self, e2e):
        self.ys = e2e[self.us, self.vs]
        return (self.ys - self.ys_).pow(2).mean()

    def get_test_loss(self, e2e):
        return (self.w_gt - e2e).view(-1).pow(2).mean()

    @FLAGS.inject
    def get_d_e2e(self, e2e, shape):
        d_e2e = torch.zeros(shape, device=device)
        d_e2e[self.us, self.vs] = self.ys - self.ys_
        d_e2e = d_e2e / len(self.ys_)
        return d_e2e

    @FLAGS.inject
    def get_cvx_opt_constraints(self, x, shape):
        A = np.zeros(shape)
        mask = np.zeros(shape)
        A[self.us, self.vs] = self.ys_
        mask[self.us, self.vs] = 1
        eps = 1.e-3
        constraints = [cvx.abs(cvx.multiply(x - A, mask)) <= eps]
        return constraints


class MatrixCompletionOld(MatrixCompletion):
    @FLAGS.inject
    def __init__(self, *, obs_path):
        self.w_gt, (self.us, self.vs), self.ys_ = torch.load(obs_path, map_location=device)


class MatrixSensing(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, gt_path, obs_path):
        self.w_gt = torch.load(gt_path, map_location=device)
        self.xs, self.ys_ = torch.load(obs_path, map_location=device)

    def get_train_loss(self, e2e):
        self.ys = (self.xs * e2e).sum(dim=-1).sum(dim=-1)
        return (self.ys - self.ys_).pow(2).mean()

    def get_test_loss(self, e2e):
        return (self.w_gt - e2e).view(-1).pow(2).mean()

    @FLAGS.inject
    def get_d_e2e(self, e2e, shape):
        d_e2e = self.xs.view(-1, *shape) * (self.ys - self.ys_).view(len(self.xs), 1, 1)
        d_e2e = d_e2e.sum(0)
        return d_e2e

    def get_cvx_opt_constraints(self, X):
        eps = 1.e-3
        constraints = []
        for x, y_ in zip(self.xs, self.ys_):
            constraints.append(cvx.abs(cvx.sum(cvx.multiply(X, x)) - y_) <= eps)
        return constraints


class MovieLens100k(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, obs_path, n_train_samples):
        (self.us, self.vs), ys_ = torch.load(obs_path, map_location=device)
        # self.ys_ = (ys_ - ys_.mean()) / ys_.std()
        self.ys_ = ys_
        self.n_train_samples = n_train_samples

    def get_train_loss(self, e2e):
        self.ys = e2e[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]]
        return (self.ys - self.ys_[:self.n_train_samples]).pow(2).mean()

    def get_test_loss(self, e2e):
        ys = e2e[self.us[self.n_train_samples:], self.vs[self.n_train_samples:]]
        return (ys - self.ys_[self.n_train_samples:]).pow(2).mean()

    @FLAGS.inject
    def get_d_e2e(self, e2e, *, shape):
        d_e2e = torch.zeros(shape, device=device)
        d_e2e[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = \
            self.ys - self.ys_[:self.n_train_samples]
        d_e2e = d_e2e / len(self.ys_)
        return d_e2e

    @FLAGS.inject
    def get_cvx_opt_constraints(self, x, *, shape):
        A = np.zeros(shape)
        mask = np.zeros(shape)
        A[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = self.ys_[:self.n_train_samples]
        mask[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = 1
        eps = 1.e-3
        constraints = [cvx.abs(cvx.multiply(x - A, mask)) <= eps]
        return constraints


@lz.main(FLAGS)
@FLAGS.inject
def main(*, depth, hidden_sizes, n_iters, problem, train_thres, _seed, _log, _writer, _info, _fs):
    prob: BaseProblem
    if problem == 'matrix-completion':
        prob = MatrixCompletion()
    elif problem == 'matrix-sensing':
        prob = MatrixSensing()
    elif problem == 'ml-100k':
        prob = MovieLens100k()
    else:
        raise ValueError

    print("hidden sizes", hidden_sizes)
    depth =len(hidden_sizes) - 1
    model = SVDLinear.SVDLinear(hidden_sizes[0], depth, M=hidden_sizes[1])
    _log.info(model)

    if FLAGS.optimizer == 'SGD':
        optimizer =  optim.SGD([{'params' : model.sigma, 'lr': FLAGS.lr}, {'params':model.thetasU}, {'params':model.thetasV}], lr=FLAGS.angle_lr)
        #optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)
    elif FLAGS.optimizer == 'GroupRMSprop':
        optimizer = GroupRMSprop(model.parameters(), FLAGS.lr, eps=1e-4)
    elif FLAGS.optimizer == 'Adam':
        optimizer = optim.Adam([{'params' : model.sigma, 'lr' : FLAGS.lr}, {'params':model.U.thetas}, {'params':model.V.thetas}], lr=FLAGS.angle_lr)
        #optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == 'cvxpy':
        cvx_opt(prob)
        return
    else:
        raise ValueError

    init_model(model)

    loss = None
    for T in range(n_iters):
        e2e = get_e2e(model)

        loss = prob.get_train_loss(e2e)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            test_loss = prob.get_test_loss(e2e)

            if  T % FLAGS.n_dev_iters == 0 or loss.item() <= train_thres: # or T % 1==0:
                U, singular_values, V = model.get_U(), model.sigma, model.get_Vt()
                schatten_norm = singular_values.pow(2.).sum()
                params_norm = U.pow(2).sum() + singular_values.pow(2).sum() + V.pow(2).sum()

                d_e2e = prob.get_d_e2e(e2e)
                full = U.t().mm(d_e2e).mm(V).abs()  # we only need the magnitude.
                n, m = full.shape

                diag = full.diag()
                mask = torch.ones_like(full, dtype=torch.int)
                mask[np.arange(min(n, m)), np.arange(min(n, m))] = 0
                off_diag = full.masked_select(mask > 0)
                _writer.add_scalar('diag/mean', diag.mean().item(), global_step=T)
                _writer.add_scalar('diag/std', diag.std().item(), global_step=T)
                _writer.add_scalar('off_diag/mean', off_diag.mean().item(), global_step=T)
                _writer.add_scalar('off_diag/std', off_diag.std().item(), global_step=T)

                grads = [param.grad.cpu().data.numpy().reshape(-1) for param in model.parameters() if param.grad is not None]
                grads = np.concatenate(grads)
                avg_grads_norm = np.sqrt(np.mean(grads**2))
                avg_param_norm = np.sqrt(params_norm.item() / len(grads))

                sing_grads = [param.grad.cpu().data.numpy().reshape(-1) for param in [ model.U.thetas, model.V.thetas] if param.grad is not None]
                sing_grads = np.concatenate(sing_grads)
                sing_avg_grads_norm = np.sqrt(np.mean(sing_grads**2))

                if isinstance(optimizer, GroupRMSprop):
                    adjusted_lr = optimizer.param_groups[0]['adjusted_lr']
                else:
                    adjusted_lr = optimizer.param_groups[0]['lr']

                _log.info(f"Iter #{T}: train = {loss.item():.3e}, test = {test_loss.item():.3e}, "
                          f"Schatten norm = {schatten_norm:.3e}, "
                          f"grad: {avg_grads_norm:.3e}, "
                          f"singular vectors' grad: {sing_avg_grads_norm:.3e}, "
                          f"lr = {adjusted_lr:.6f}")

                product = get_e2e(model).detach()
                u, s, v = torch.svd(product)
                #_log.info(f"torch.svd SIGMA = {s.topk(10)}")
                _log.info(f"NONZERO SIGMA #  = {torch.count_nonzero(torch.round(s))}")

                _writer.add_scalar('loss/train', loss.item(), global_step=T)
                _writer.add_scalar('loss/test', test_loss, global_step=T)
                _writer.add_scalar('Schatten_norm', schatten_norm, global_step=T)
                _writer.add_scalar('norm/grads', avg_grads_norm, global_step=T)
                _writer.add_scalar('norm/params', avg_param_norm, global_step=T)

                for i in range(FLAGS.n_singulars_save):
                    _writer.add_scalar(f'singular_values/{i}', singular_values[0,i], global_step=T)

                torch.save(e2e, _fs.resolve("$LOGDIR/final.npy"))
                if loss.item() <= train_thres:
                    break

        optimizer.step()


    _log.info(f"train loss = {loss.item()}. test loss = {test_loss.item()}")

    _, singular_values, _ = model.detach_SVD()
    singular_values, indices = torch.abs(singular_values).sort(descending=True)
    if singular_values.size(1) < 10:
        singular_values = singular_values.t()
   
    _log.info(f"model SIGMA = {singular_values.topk(10)}")
        
    _log.info(f"NONZERO SIGMA #  = {torch.count_nonzero(torch.round(singular_values))}")

    product = get_e2e(model).detach()
    u, s, v = torch.svd(product)
    _log.info(f"torch.svd SIGMA = {s.topk(10)}")
    _log.info(f"NONZERO SIGMA #  = {torch.count_nonzero(torch.round(s))}")

    #_log.info(f"MODEL  U: {model.U.getU()[indices]}")
    #_log.info(f"torch svd  U: {u}")



if __name__ == '__main__':
    main()
