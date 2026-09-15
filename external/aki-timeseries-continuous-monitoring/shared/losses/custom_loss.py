import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class Cosine_Loss(pl.LightningModule):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_similarity = nn.CosineEmbeddingLoss(reduction="mean")
        self.condition = torch.Tensor([1])
        self.num_class = num_class

    def forward(self, logits, labels):
        loss_cross = self.cross_entropy(logits, labels)
        loss_cosine = self.cosine_similarity(
            logits,
            F.one_hot(labels, num_classes=self.num_class),
            self.condition.type_as(loss_cross)
        )
        loss = loss_cosine + 0.1 * loss_cross
        return loss

class WeightedKappaLoss(pl.LightningModule):
    """
    Implements Weighted Kappa Loss. Weighted Kappa Loss was introduced in the
    [Weighted kappa loss function for multi-class classification
    of ordinal data in deep learning]
    (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems. The loss
    value lies in $[-\\infty, \\log 2]$, where $\\log 2$ means the random prediction
    Usage: loss_fn = WeightedKappaLoss(num_classes = NUM_CLASSES)
    """
    def __init__(
        self,
        num_classes: int,
        mode: Optional[str] = 'quadratic',
        name: Optional[str] = 'cohen_kappa_loss',
        epsilon: Optional[float]= 1e-10
    ):

        """Creates a `WeightedKappaLoss` instance.
            Args:
            num_classes: Number of unique classes in your dataset.
            weightage: (Optional) Weighting to be considered for calculating
                kappa statistics. A valid value is one of
                ['linear', 'quadratic']. Defaults to 'quadratic'.
            name: (Optional) String name of the metric instance.
            epsilon: (Optional) increment to avoid log zero,
                so the loss will be $ \\log(1 - k + \\epsilon) $, where $ k $ lies
                in $ [-1, 1] $. Defaults to 1e-10.
            Raises:
            ValueError: If the value passed for `weightage` is invalid
                i.e. not any one of ['linear', 'quadratic']
            """

        super(WeightedKappaLoss, self).__init__()
        self.num_classes = num_classes
        if mode == 'quadratic':
            self.y_pow = 2
        if mode == 'linear':
            self.y_pow = 1

        self.epsilon = epsilon

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).type_as(y_pred)
        y_true = y[y_true]

        y_true = y_true.float()

        repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).type_as(y_pred)
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / (self.epsilon + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(torch.reshape(hist_rater_a, [num_classes, 1]),
                                    torch.reshape(hist_rater_b, [1, num_classes]))
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.epsilon)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)
