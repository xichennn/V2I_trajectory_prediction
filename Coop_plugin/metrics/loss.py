import torch
import torch.nn as nn
from Coop_plugin.metrics.metric import min_ade, traj_nll

class NLLloss(nn.Module):
    """
    MTP loss modified to include variances. Uses MSE for mode selection. Can also be used with
    Multipath outputs, with residuals added to anchors.
    """
    def __init__(self, alpha=0.2, use_variance=True, device='cpu'):
        """
        Initialize MSMA loss
        :param args: Dictionary with the following (optional) keys
            use_variance: bool, whether or not to use variances for computing regression component of loss,
                default: False
            alpha: float, relative weight assigned to classification component, compared to regression component
                of loss, default: 1
        """
        super(NLLloss, self).__init__()
        self.use_variance = use_variance
        self.alpha = alpha
        self.device = device

    def forward(self, y_pred, y_true, log_probs):
        """
        params:
        :y_pred: [num_nodes, num_modes, op_len, 2]
        :y_true: [num_nodes, op_len, 2]
        :log_probs: probability for each mode [N_B, N_M]
        where N_B is batch_size, N_M is num_modes, op_len is target_len
        """


        num_nodes = y_true.shape[0]
        l2_norm = (torch.norm(y_pred - y_true.unsqueeze(1), p=2, dim=-1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=1)
        pred_best = y_pred[torch.arange(num_nodes), best_mode, :, :]


        loss_cls = (-log_probs[torch.arange(num_nodes).to(self.device), best_mode].squeeze()).mean() #[N_B]

        loss_reg = (torch.norm(pred_best-y_true, p=2, dim=-1)).mean()


        loss = loss_reg + self.alpha * loss_cls

        return loss