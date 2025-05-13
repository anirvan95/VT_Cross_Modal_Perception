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


def bayes_fusion(prior_dist_params, like_dist_params, weight=1):
    # if weight < 1 - prior model dominates
    # if weight > 1 - likelihood model dominates
    # weight = 1 - no change

    prior_mean = prior_dist_params[:, :, 0:1]
    prior_logsigma = prior_dist_params[:, :, 1:2]

    like_mean = like_dist_params[:, :, 0:1]
    like_logsigma = like_dist_params[:, :, 1:2]

    prior_variance = torch.exp(prior_logsigma) ** 2
    like_variance = (torch.exp(like_logsigma) ** 2) / weight

    normalization = prior_variance + like_variance

    posterior_mean = (like_variance * prior_mean + prior_variance * like_mean) / normalization
    posterior_variance = (like_variance * prior_variance) / normalization

    return torch.cat([posterior_mean, torch.log(torch.sqrt(posterior_variance))], dim=-1)
