import pytorch_lightning as pl
import sys
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, AUROC, AveragePrecision, Precision, Recall, CohenKappa, F1Score, MeanMetric, MaxMetric
from torch.utils.data import DataLoader
from typing import Any, Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.losses import Cosine_Loss, WeightedKappaLoss
from shared.optimizers import SAM


class AKI_Simple_TrainModule(pl.LightningModule):
    
    def __init__(
        self, 
        backbone_model: torch.nn.Module,
        num_class:int,
        ordinal_class: bool = False
    ) -> None:
        super().__init__()
        
        # ensure init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, 
            ignore=["backbone_model"]
        )
        
        # backbone_model
        self.backbone_model = backbone_model
        
        # loss function
        if ordinal_class:
            self.criterion = WeightedKappaLoss(num_classes=num_class)
        else:
            self.criterion = Cosine_Loss(num_class=num_class)
        
        # metric objects for calculating and averaging metrics across batches
        
        ## auroc
        self.train_auroc = AUROC(task="multiclass", num_classes=num_class)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_class)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_class)
        ## auprc
        self.train_auprc = AveragePrecision(task="multiclass", num_classes=num_class)
        self.val_auprc = AveragePrecision(task="multiclass", num_classes=num_class)
        self.test_auprc = AveragePrecision(task="multiclass", num_classes=num_class)
        ## accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_class)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_class)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_class)
        ## f1 score
        self.train_f1 = F1Score(task="multiclass", num_classes=num_class, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_class, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_class, average="macro")
        ## cohen kappa
        self.train_cohen = CohenKappa(task="multiclass", weights="quadratic", num_classes=num_class)
        self.val_cohen = CohenKappa(task="multiclass", weights="quadratic", num_classes=num_class)
        self.test_cohen = CohenKappa(task="multiclass", weights="quadratic", num_classes=num_class)
        ## precision
        self.train_precision = Precision(task="multiclass", average="macro", num_classes=num_class)
        self.val_precision = Precision(task="multiclass", average="macro", num_classes=num_class)
        self.test_precision = Precision(task="multiclass", average="macro", num_classes=num_class)
        ## recall
        self.train_recall = Recall(task="multiclass", average="macro", num_classes=num_class)
        self.val_recall = Recall(task="multiclass", average="macro", num_classes=num_class)
        self.test_recall = Recall(task="multiclass", average="macro", num_classes=num_class)
        
        # averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # traking best 
        self.val_acc_best = MaxMetric()
        self.val_auroc_best = MaxMetric()
        self.val_auprc_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.val_cohen_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        
        self.test_acc_best = MaxMetric()
        self.test_auroc_best = MaxMetric()
        self.test_auprc_best = MaxMetric()
        self.test_f1_best = MaxMetric()
        self.test_cohen_best = MaxMetric()
        self.test_precision_best = MaxMetric()
        self.test_recall_best = MaxMetric()
        
    def forward(
        self, 
        meta:torch.FloatTensor, 
        x:torch.FloatTensor, 
    ):
        logits = self.backbone_model(meta, x)
        return logits
    
    def on_train_start(self) -> None:
        
        self.val_loss.reset()
        
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_f1.reset()
        self.val_cohen.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        
        self.val_acc_best.reset()
        self.val_auroc_best.reset()
        self.val_auprc_best.reset()
        self.val_f1_best.reset()
        self.val_cohen_best.reset()
        self.val_precision_best.reset()
        self.val_recall_best.reset()
        
        return None
    
    def model_step(self, batch:Any):
        
        time_series_data = batch["data"] 
        demo_data = batch["meta"]
        label = batch["label"]
        
        logit = self.forward(demo_data, time_series_data)
        
        loss = self.criterion(logit, label)
        
        return loss, logit, label
        
    
    def training_step(self, batch:Dict[str, torch.Tensor], batch_idx:int):
        
        loss, logits, label = self.model_step(batch)
        
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(logits, label)
        self.train_auroc(logits, label)
        self.train_auprc(logits, label)
        self.train_f1(logits, label)
        self.train_cohen(logits, label)
        self.train_precision(logits, label)
        self.train_recall(logits, label)
        
        logs = {
            "train/loss":self.train_loss,
            "train/acc":self.train_acc,
            "train/auroc":self.train_auroc,
            "train/auprc":self.train_auprc,
            "train/f1":self.train_f1,
            "train/cohen":self.train_cohen,
            "train/precision":self.train_precision,
            "train/recall":self.train_recall,
        }
        
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False)
        
        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.trainer.strategy.barrier()
    
    def validation_step(self, batch:Dict[str, torch.Tensor], batch_idx:int):
        
        loss, logits, label = self.model_step(batch)
        
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(logits, label)
        self.val_auroc(logits, label)
        self.val_auprc(logits, label)
        self.val_f1(logits, label)
        self.val_cohen(logits, label)
        self.val_precision(logits, label)
        self.val_recall(logits, label)
        
        logs = {
            "val/loss":self.val_loss,
            "val/acc":self.val_acc,
            "val/auroc":self.val_auroc,
            "val/auprc":self.val_auprc,
            "val/f1":self.val_f1,
            "val/cohen":self.val_cohen,
            "val/precision":self.val_precision,
            "val/recall":self.val_recall,
        }
        
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        
        auroc = self.val_auroc.compute()
        self.val_auroc_best(auroc)
        
        auprc = self.val_auprc.compute()
        self.val_auprc_best(auprc)
        
        f1 = self.val_f1.compute()
        self.val_f1_best(f1)
        
        cohen = self.val_cohen.compute()
        self.val_cohen_best(cohen)
        
        precision = self.val_precision.compute()
        self.val_precision_best(precision)
        
        recall = self.val_recall.compute()
        self.val_recall_best(recall)
        
        logs = {
            "val/acc_best":self.val_acc_best.compute(),
            "val/auroc_best":self.val_auroc_best.compute(),
            "val/auprc_best":self.val_auprc_best.compute(),
            "val/f1_best":self.val_f1_best.compute(),
            "val/cohen_best":self.val_cohen_best.compute(),
            "val/precision_best":self.val_precision_best.compute(),
            "val/recall_best":self.val_recall_best.compute(),
        }
        
        self.log_dict(logs, sync_dist=True, prog_bar=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx:int) -> None:
        
        loss, logits, label = self.model_step(batch)
        
        # update and log metrics
        self.test_loss(loss),
        self.test_acc(logits, label)
        self.test_auroc(logits, label)
        self.test_auprc(logits, label)
        self.test_f1(logits, label)
        self.test_cohen(logits, label)
        self.test_precision(logits, label)
        self.test_recall(logits, label)
        
        logs = {
            "test/loss":self.test_loss,
            "test/acc":self.test_acc,
            "test/auroc":self.test_auroc,
            "test/auprc":self.test_auprc,
            "test/f1":self.test_f1,
            "test/cohen":self.test_cohen,
            "test/precision":self.test_precision,
            "test/recall":self.test_recall,
        }
        
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self) -> None:
        
        acc = self.test_acc.compute()
        self.test_acc_best(acc)
        
        auroc = self.test_auroc.compute()
        self.test_auroc_best(auroc)
        
        auprc = self.test_auprc.compute()
        self.test_auprc_best(auprc)
        
        f1 = self.test_f1.compute()
        self.test_f1_best(f1)
        
        cohen = self.test_cohen.compute()
        self.test_cohen_best(cohen)
        
        precision = self.test_precision.compute()
        self.test_precision_best(precision)
        
        recall = self.test_recall.compute()
        self.test_recall_best(recall)
        
        logs = {
            "test/acc_best":self.test_acc_best.compute(),
            "test/auroc_best":self.test_auroc_best.compute(),
            "test/auprc_best":self.test_auprc_best.compute(),
            "test/f1_best":self.test_f1_best.compute(),
            "test/cohen_best":self.test_cohen_best.compute(),
            "test/precision_best":self.test_precision_best.compute(),
            "test/recall_best":self.test_recall_best.compute(),
        }
        
        self.log_dict(logs, sync_dist=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        time_series_data = batch["data"] 
        demo_data = batch["meta"]
        
        logits = self.forward(demo_data, time_series_data)
        return logits
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.001
        )
        # optimizer = SAM(
        #     params=self.parameters(),
        #     base_optimizer=base_optimizer,
        #     rho=0.05,
        #     adaptive=True,
        #     lr=0.001
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=0.001
        )
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
        
    # def optimizer_step(
    #     self,
    #     epoch: int,
    #     batch_idx: int,
    #     optimizer,
    #     optimizer_closure,
    #     optimizer_idx: int = 0,
    #     on_tpu: bool = False,
    #     using_native_amp: bool = False,
    #     using_lbfgs: bool = False,
    # ):
    #     # SAM Optimizer의 두 단계 수행
    #     optimizer_closure()
    #     optimizer.first_step(zero_grad=True)
    #     optimizer_closure()
    #     optimizer.second_step(zero_grad=True)