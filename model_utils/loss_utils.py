import torch
import torch.nn.functional as F
import numpy as np


class SphereDistance(torch.nn.Module):
    """Contrastive Classifier.
    Calculates the distance between two random vectors, and returns an exponential transformation of it,
    which can be interpreted as the logits for the two vectors being different.
    p : Probability of x1 and x2 being different
    p = 1 - exp( -dist(x1,x2) )
    """

    def __init__(
        self,
        distance: torch.nn.Module,
    ):
        """
        Args:
            distance : A Pytorch module which takes two (batches of) vectors and returns a (batch of)
                positive number.
        """
        super().__init__()

        self.distance = distance

        self.eps = 1e-10

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:

        # Compute distance
        dists = self.distance(x1, x2)
        # if len(dists.shape) == 3:
        #     dists = torch.mean(dists, axis=(-1, -2)).unsqueeze(1)
        return dists#torch.exp(-dists)


class LocalContrastiveLoss(torch.nn.Module):

    def __init__(self, distance: torch.nn.Module):
        """
        Args:
            distance : A Pytorch module which takes two (batches of) vectors and returns a (batch of)
                positive number.
        """
        super().__init__()

        self.distance = SphereDistance(distance)

    def forward(self, query, suspect_logits, reduction=True, regularization=False):

        #K = suspect_logits.shape[1]
        #query = query.repeat(1, K, 1)

        #suspect_logits = F.normalize(suspect_logits, p=2, dim=-1)
        #query = F.normalize(query, p=2, dim=-1)
        distance = torch.pow(self.distance(query, suspect_logits), 1)
        if reduction:
            contrast_loss = torch.mean(distance.sum(1))
        else:
            contrast_loss = distance#.view(-1)


        if regularization and reduction:

            std_x = torch.sqrt(query.var(dim=0) + 0.0001)
            std_y = torch.sqrt(suspect_logits.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
            contrast_loss = contrast_loss + std_loss
        return contrast_loss


class NeuralTransformationLoss(torch.nn.Module):
    def __init__(self,temperature=1, hidden_dim=None, distance_type="cosine"):
        super().__init__()
        self.temp = temperature
        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.out_layer = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim, bias=False), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.distance_type = distance_type
        self.eps = 1e-10

    def forward(self, z, reduction=False):

        if self.hidden_dim:
            z = self.out_layer(z)

        if self.distance_type == "cosine":
            z = F.normalize(z, p=2, dim=-1) # B, K, d
            z_ori = z[:, 0]  # n,z
            z_trans = z[:, 1:]  # n,k-1, z
            batch_size, num_trans, z_dim = z.shape

            sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        else:
            z_ori = z[:, 0]  # n,z
            z_trans = z[:, 1:]  # n,k-1, z
            batch_size, num_trans, z_dim = z.shape
            sim_matrix = torch.exp(-torch.cdist(z, z))

        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0)\
                .to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

        if self.distance_type == "cosine":
            pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp) # n,k-1
        else:
            pos_sim = torch.exp(-torch.cdist(z_trans, z_ori.unsqueeze(1))).squeeze(-1)

        K = num_trans - 1
        scale = 1 / np.abs(np.log(1.0 / K))

        loss_tensor = (torch.log(trans_matrix+self.eps) - torch.log(pos_sim+self.eps))# * scale

        if reduction:
            score = torch.mean(loss_tensor.sum(1))
            return score
        else:
            return loss_tensor.sum(1)


class CosineDistance(torch.nn.Module):
    r"""Returns the cosine distance between :math:`x_1` and :math:`x_2`, computed along dim."""

    def __init__(
        self,
        dim: int = 1,
        keepdim: bool = True,
        hidden_dim=None,
    ) -> None:

        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10
        self.hidden_dim = hidden_dim
        if hidden_dim is not None:
            self.weight = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        # Cosine of angle between x1 and x2
        if self.hidden_dim is not None:
            x2 = self.weight(x2)

        cos_sim = F.cosine_similarity(x1, x2, self.dim, self.eps)
        dist = -torch.log((1 + cos_sim) / 2)/0.1

        if self.keepdim:
            dist = dist.unsqueeze(dim=self.dim)
        return dist


class LpDistance(torch.nn.Module):
    r"""Returns the Lp norm between :math:`x_1` and :math:`x_2`, computed along dim."""

    def __init__(
        self,
        p: int = 2,
        dim: int = 1,
        keepdim: bool = False,
    ) -> None:

        super().__init__()
        self.dim = int(dim)
        self.p = int(p)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        # Lp norm between x1 and x2
        dist = torch.norm(x2 - x1, p=self.p, dim=self.dim, keepdim=self.keepdim)

        return dist


class NeuralDistance(torch.nn.Module):
    """Neural Distance
    Transforms two vectors into a single positive scalar, which can be interpreted as a distance.
    """

    def __init__(self, rep_dim: int, layers: int = 1) -> None:

        super().__init__()

        rep_dim = int(rep_dim)
        layers = int(layers)
        if layers < 1:
            raise ValueError("layers>=1 is required")
        net_features_dim = np.linspace(rep_dim, 1, layers + 1).astype(int)

        net = []
        for i in range(layers):
            net.append(torch.nn.Linear(net_features_dim[i], net_features_dim[i + 1]))
            if i < (layers - 1):
                net.append(torch.nn.ReLU())

        net.append(torch.nn.Softplus(beta=1))

        self.net = torch.nn.Sequential(*net)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:

        out = self.net(x2 - x1)

        return out


class InfoNCE(torch.nn.Module):
    """Contrastive Classifier.
    Calculates the distance between two random vectors, and returns an exponential transformation of it,
    which can be interpreted as the logits for the two vectors being different.
    p : Probability of x1 and x2 being different
    p = 1 - exp( -dist(x1,x2) )
    """

    def __init__(
        self,
        distance: torch.nn.Module,
    ):
        """
        Args:
            distance : A Pytorch module which takes two (batches of) vectors and returns a (batch of)
                positive number.
        """
        super().__init__()

        self.distance = distance
        self.eps = 1e-10

    def forward(
        self,
        query: torch.Tensor,
        pos_logit: torch.Tensor,
        neg_logits: torch.Tensor=None,
    ) -> torch.Tensor:

        # Compute distance
        pos_logit = torch.exp(-self.distance(query, pos_logit))

        # Probability of the two embeddings being equal: exp(-dist)
        log_prob_equal = torch.log(pos_logit+self.eps)

        if neg_logits is not None:
            # Computation of log_prob_different
            prob_different = 0
            for i in range(neg_logits.shape[1]):
                neg_logit = torch.exp(-self.distance(query, neg_logits[:,i]))
                prob_different += neg_logit#torch.clamp(1 - neg_logit, self.eps, 1)
            prob_different += pos_logit
            log_prob_different = torch.log(prob_different+self.eps)
            # prob_different = torch.clamp(1 - torch.exp(-neg_dists), self.eps, 1)
            # log_prob_different = torch.log(prob_different)

            logits_different = -(log_prob_equal - log_prob_different)

            return logits_different
        else:
            return -log_prob_equal


class CosineDistance(torch.nn.Module):
    r"""Returns the cosine distance between :math:`x_1` and :math:`x_2`, computed along dim."""

    def __init__(
        self,
        dim: int = 1,
        keepdim: bool = False,
        hidden_dim=None,
    ) -> None:

        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10
        self.hidden_dim = hidden_dim
        if hidden_dim is not None:
            self.weight = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        # Cosine of angle between x1 and x2
        if self.hidden_dim is not None:
            x2 = self.weight(x2)

        cos_sim = F.cosine_similarity(x1, x2, self.dim, self.eps)
        #dist = -torch.log((1 + cos_sim) / 2)
        dist = -cos_sim

        if self.keepdim:
            dist = dist.unsqueeze(dim=self.dim)
        return dist