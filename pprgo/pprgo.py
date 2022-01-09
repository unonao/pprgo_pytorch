import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from .pytorch_utils import MixedDropout, MixedLinear


class PPRGoMLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()

        fcs = [MixedLinear(num_features, hidden_size, bias=False)]
        for i in range(nlayers - 2):
            fcs.append(nn.Linear(hidden_size, hidden_size, bias=False))
        fcs.append(nn.Linear(hidden_size, num_classes, bias=False))
        self.fcs = nn.ModuleList(fcs)

        self.drop = MixedDropout(dropout)

    def forward(self, X):
        embs = self.drop(X)
        embs = self.fcs[0](embs)
        for fc in self.fcs[1:]:
            embs = fc(self.drop(F.relu(embs)))
        return embs


class PPRGo(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes, hidden_size, nlayers, dropout)

    def forward(self, X, ppr_scores, ppr_idx):
        logits = self.mlp(X)


        """
        https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        X: PPRの対象となる頂点 x 属性数
        ppr_scores: X に対応するpprのスコア
        ppr_idx: source となる頂点

        XをNNにかけて出てきた特徴量を、pprで重み付けし、対応するsourceにsumで合わせる
        """
        propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        return propagated_logits


class MultiPPRGo(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes, hidden_size, nlayers, dropout)

    def forward(self, X, ppr_scores, ppr_idx):
        logits = self.mlp(X)

        """
        https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        X: PPRの対象となる頂点 x 属性数
        ppr_scores: X に対応するpprのスコア
        ppr_idx: source となる頂点

        XをNNにかけて出てきた特徴量を、pprで重み付けし、対応するsourceにsumで合わせる
        """
        propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        return propagated_logits



class MultiPPRGo(nn.Module):
    def __init__(self, num_features, num_classes, num_ppr, hidden_size, nlayers, dropout):
        super().__init__()
        self.num_ppr = num_ppr
        self.mlp = PPRGoMLP(num_features, hidden_size, hidden_size, nlayers, dropout)
        self.linear_squeeze = nn.Linear(num_ppr, 1)
        self.linear_head = nn.Linear(hidden_size, num_classes)

    def forward(self, X, ppr_scores_list, ppr_idx_list):
        logits = self.mlp(X)
        propagated_logits_list = []
        for i in range(self.num_ppr):
            ppr_scores = ppr_scores_list[i]
            ppr_idx = ppr_idx_list[i]
            propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
            propagated_logits_list.append(propagated_logits)

        x_3d = torch.stack(propagated_logits_list, dim=3)
        x_2d = self.linear_squeeze(x_3d).squeeze()
        x_2d = self.linear_head(x_2d)
        return x_2d
