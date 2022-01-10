import torch

from .pytorch_utils import matrix_to_torch


class PPRDataset(torch.utils.data.Dataset):
    def __init__(self, attr_matrix_all, ppr_matrix, indices, labels_all=None):
        self.attr_matrix_all = attr_matrix_all  # (all_node_num, attr_num(feature_num))
        self.ppr_matrix = ppr_matrix            # (train_node_num, all_node_num)
        self.indices = indices
        self.labels_all = torch.tensor(labels_all)
        self.cached = {}

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        # idx is a list of indices(idx は、ppr_matrixにおける行番号のリスト)
        key = idx[0]
        if key not in self.cached:
            ppr_matrix = self.ppr_matrix[idx]  # (num(idx), all_node_num)
            source_idx, neighbor_idx = ppr_matrix.nonzero()  # 非ゼロの要素のインデックス ppr_matrixの source_idx[i]行目、neighbor_idx[i]列目に非ゼロの要素がある
            ppr_scores = ppr_matrix.data  # 非ゼロの実際の値  (num(ppr_matrix.nonzero()))

            attr_matrix = matrix_to_torch(self.attr_matrix_all[neighbor_idx])  # (num(ppr_matrix.nonzero()), 属性数)
            ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)  # (num(ppr_matrix.nonzero()))
            source_idx = torch.tensor(source_idx, dtype=torch.long)     # (num(ppr_matrix.nonzero()))

            if self.labels_all is None:
                labels = None
            else:
                labels = self.labels_all[self.indices[idx]]
            self.cached[key] = ((attr_matrix, ppr_scores, source_idx), labels)
        return self.cached[key]


class SAPPRDataset(torch.utils.data.Dataset):
    """
    ppr_matrix を複数受け取って、それぞれのppr_matrixに応じたデータを返すようにする。
    """

    def __init__(self, attr_matrix_all, ppr_matrix_list, indices, labels_all=None):
        self.attr_matrix_all = attr_matrix_all
        self.ppr_matrix_list = ppr_matrix_list
        self.indices = indices
        self.labels_all = torch.tensor(labels_all)
        self.cached = {}

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        # idx is a list of indices(idx は、ppr_matrixにおける行番号のリスト)
        key = idx[0]
        if key not in self.cached:
            attr_matrix_list = []
            ppr_matrix_list = []
            source_idx_list = []
            for ppri in range(len(self.ppr_matrix_list)):
                ppr_matrix = self.ppr_matrix_list[ppri][idx]  # (num(idx), all_node_num)
                source_idx, neighbor_idx = ppr_matrix.nonzero()  # 非ゼロの要素のインデックス ppr_matrixの source_idx[i]行目、neighbor_idx[i]列目に非ゼロの要素がある
                ppr_scores = ppr_matrix.data  # 非ゼロの実際の値  (num(ppr_matrix.nonzero()))

                attr_matrix = matrix_to_torch(self.attr_matrix_all[neighbor_idx])  # (num(ppr_matrix.nonzero()), 属性数)
                ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)  # (num(ppr_matrix.nonzero()))
                source_idx = torch.tensor(source_idx, dtype=torch.long)     # (num(ppr_matrix.nonzero()))
                attr_matrix_list.append(attr_matrix)
                ppr_matrix_list.append(ppr_scores)
                source_idx_list.append(source_idx)

            if self.labels_all is None:
                labels = None
            else:
                labels = self.labels_all[self.indices[idx]]

            self.cached[key] = ((attr_matrix_list, ppr_matrix_list, source_idx_list), labels)

        return self.cached[key]
