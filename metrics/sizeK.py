from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class Size(Metric):

    def __init__(self,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(Size, self).__init__(dist_sync_on_step=dist_sync_on_step,
                                 process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor) -> None:
        #pred: [K, N, T, 4]
        loc, scale = pred.chunk(2, dim=-1)
        z_score = 1.645
        eplis_radius = z_score * scale
        area = torch.pi * eplis_radius[...,0] * eplis_radius[...,1]
        self.sum += torch.sum(area)/(area.shape[0]*area.shape[2])
        self.count += pred.size(1)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
