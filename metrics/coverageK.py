from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class Coverage(Metric):

    def __init__(self,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(Coverage, self).__init__(dist_sync_on_step=dist_sync_on_step,
                                 process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('coverage_rate', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        #pred: [K, N, T, 4], target: [N, T, 2]
        loc, scale = pred.chunk(2, dim=-1)
        # prediction_intervals: [K, N, 50, 4], 4->lower_x, upper_x, lower_y, upper_y
        # Z-score for a 90% confidence interval
        z_score = 1.645

        # Calculate the confidence interval
        upper_bound = loc + z_score * scale
        lower_bound = loc - z_score * scale

        condition = torch.ones(pred.shape[0], pred.shape[1], pred.shape[2],
                             dtype=torch.bool, device=pred.device) #[K, N, T]
        condition_c = condition.clone()
        for k in range(pred.shape[0]):
            for i in range(2):
                condition_c[k] &= torch.bitwise_and(target[...,i] >= lower_bound[k,:,:,i], target[...,i] <= upper_bound[k,:,:,i])
            
        self.coverage_rate += torch.max(torch.sum(condition_c, dim=(1,2)), dim=0)[0]/condition_c.shape[2]
        self.count += pred.size(1)

    def compute(self) -> torch.Tensor:
        return self.coverage_rate / self.count
