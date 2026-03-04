import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class DistillationLoss(nn.Module):
    def __init__(self, T=1.5):
        super().__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits):
        student = torch.log_softmax(student_logits / self.T, dim=1)
        teacher = torch.softmax(teacher_logits / self.T, dim=1)
        loss = -1 * torch.mul(teacher, student).sum()
        
        return loss


class MultiLabelDistillationLoss(nn.Module):
    def __init__(self, reduction="batchmean", eps=1e-3):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction="none")

    def forward(self, student_logits, teacher_logits):
        N, C = student_logits.shape
        student = torch.sigmoid(student_logits)
        teacher = torch.sigmoid(teacher_logits)
        
        # Clamp values for numerical stability
        student = torch.clamp(student, min=self.eps, max=1-self.eps)
        teacher = torch.clamp(teacher, min=self.eps, max=1-self.eps)
        
        # Compute KL divergence for both positive and negative probabilities
        loss = self.criterion(torch.log(student), teacher) + self.criterion(torch.log(1 - student), 1 - teacher)

        # Apply reduction
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "batchmean":
            loss = loss.sum() / N
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise AttributeError("Invalid reduction method")
        
        if torch.isnan(loss):
            loss = 1e-6
        
        return loss