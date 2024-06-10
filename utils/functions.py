import torch
from torch.autograd import Function


class STHeaviside(Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.zeros(x.size()).type_as(x)
        y[x >= 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def bayes_fusion(trans_dist_params, meas_dist_params):
    trans_mean = trans_dist_params[:, :, 0:1]
    trans_logsigma = trans_dist_params[:, :, 1:2]
    meas_mean = meas_dist_params[:, :, 0:1]
    mean_logsigma = meas_dist_params[:, :, 1:2]
    trans_variance = torch.exp(trans_logsigma) ** 2
    meas_variance = torch.exp(mean_logsigma) ** 2
    normalization = trans_variance + meas_variance
    posterior_mean = (meas_variance * trans_mean + trans_variance * meas_mean) / normalization
    posterior_variance = (meas_variance * trans_variance) / normalization

    return torch.cat([posterior_mean, torch.log(torch.sqrt(posterior_variance))], dim=-1)
