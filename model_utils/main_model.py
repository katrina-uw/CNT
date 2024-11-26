from torch import nn
import torch
from model_utils.feature_encoder import TCNFeatureEncoder as FeatureEncoder
from model_utils.feature_encoder import TCNFeatureEncoder_separate as FeatureEncoder_separate
from model_utils.loss_utils import NeuralTransformationLoss, LocalContrastiveLoss, LpDistance, InfoNCE, CosineDistance
from model_utils.graph_constructor import construct_cosine_graph
from model_utils.SeqNets import SeqNets


class MainModel(nn.Module):

    def __init__(self, config):
        super(MainModel, self).__init__()
        self.target_dims = config.target_dims
        self.loss_type = config.loss_type
        self.n_transformations = config.n_transformations
        self.target_dims = config.target_dims
        self.topK = config.topK
        self.trans_type = config.trans_type
        self.target_dims = config.target_dims
        self.hidden_dim = config.hidden_dim
        if self.target_dims is not None:
            config.feature_dim = len(self.target_dims)

        if "OCC" in self.loss_type:
            self.distance = LpDistance(dim=2)
            self.occ_loss_fn = LocalContrastiveLoss(distance=self.distance)

        if "NTL" in self.loss_type:
            self.ntl_loss_fn = NeuralTransformationLoss(temperature = config.temperature, distance_type="cosine")

        if "infoNCE" in self.loss_type:
            self.distance = LpDistance(dim=2)
            #self.distance = CosineDistance(dim=2)
            self.infoNCE_loss_fn = InfoNCE(distance=self.distance)

        if "featureNTL" in self.loss_type:
            self.feature_encoder = FeatureEncoder(config)
            self.linear_transforms = torch.nn.ModuleList()
            for _ in range(self.n_transformations):
                self.linear_transforms.append(torch.nn.Sequential(torch.nn.Linear(config.hidden_dim, config.hidden_dim), \
                                                                  torch.nn.ReLU(), torch.nn.Linear(config.hidden_dim, config.hidden_dim),\
                                                                  torch.nn.ReLU(), torch.nn.Linear(config.hidden_dim, config.hidden_dim)))
        elif "rawNTL" in self.loss_type:
            model = SeqNets()
            self.enc, self.trans = model._make_nets(config)
        else:
            if "stopping_gradient" in self.loss_type:
                self.feature_encoder = FeatureEncoder_separate(config)
            else:
                self.feature_encoder = FeatureEncoder(config)

    def forward(self, x):
        if self.target_dims is not None:
            x = x[:, :, self.target_dims]
            assert len(x.shape) == 3
        if "featureNTL" in self.loss_type:

            context_graph_state, suspect_graph_state = self.feature_encoder(x)

            suspect_graph_state = suspect_graph_state.sum(1)  # B, c, d
            context_graph_state = context_graph_state.sum(1) # B, c, d
            suspect_graph_states = [suspect_graph_state]
            for k in range(self.n_transformations):
                suspect_graph_states.append(self.linear_transforms[k](suspect_graph_state))
            suspect_graph_states = torch.cat([s.unsqueeze(1) for s in suspect_graph_states], axis=1)
        elif "rawNTL" in self.loss_type:
            x = x.transpose(1, 2)
            x_T = torch.empty(x.shape[0], self.n_transformations, x.shape[1], x.shape[2]).to(x)
            for i in range(self.n_transformations):
                mask = self.trans[i](x)

                if self.trans_type == 'forward':
                    x_T[:, i] = mask
                elif self.trans_type == 'mul':
                    mask = torch.sigmoid(mask)
                    x_T[:, i] = mask * x
                elif self.trans_type == 'residual':
                    x_T[:, i] = mask + x
            x_cat = torch.cat([x.unsqueeze(1), x_T], 1)
            zs = self.enc[0](x_cat.reshape(-1, x.shape[1], x.shape[2]))
            suspect_graph_states = zs.reshape(x.shape[0], self.n_transformations + 1, self.hidden_dim)
            suspect_graph_state = None
            context_graph_state = None
        else:
            context_graph_state, suspect_graph_state = self.feature_encoder(x)
            suspect_graph_states = None

        return context_graph_state, suspect_graph_state, suspect_graph_states

    def get_infoNCE_loss(self, context_graph_state, suspect_graph_state, reduction=True):
        k = 12
        negative_context_graph_states = []
        batch_size = context_graph_state.shape[0]
        for i in range(k):
            index = torch.randint(low=0, high=batch_size, size=(batch_size,))
            negative_context_graph_states.append(context_graph_state[index])
        negative_context_graph_states = torch.cat([s.unsqueeze(1) for s in negative_context_graph_states], axis=1)

        contrast_loss = self.infoNCE_loss_fn(query=suspect_graph_state, pos_logit=context_graph_state, neg_logits=negative_context_graph_states)
        if reduction:
            return {"infoNCE": torch.mean(contrast_loss)}
        else:
            return {"infoNCE": contrast_loss}


    def get_contextual_OCC_loss(self, context_graph_state, suspect_graph_state, reduction=True):
        if "reg" in self.loss_type:
            contrast_loss = self.occ_loss_fn(context_graph_state, suspect_graph_state, reduction=reduction, regularization=True)
        elif "stopping_gradient" in self.loss_type:
            contrast_loss = self.occ_loss_fn(context_graph_state, suspect_graph_state.detach(), reduction=reduction, regularization=False)
            contrast_loss += self.occ_loss_fn(context_graph_state.detach(), suspect_graph_state, reduction=reduction, regularization=False)
            contrast_loss /= 2
        else:
            contrast_loss = self.occ_loss_fn(context_graph_state, suspect_graph_state, reduction=reduction, regularization=False)
        # if reduction:
        #     return contrast_loss
        # else:
        return {"OCC": contrast_loss}

    def get_NTL_loss(self, suspect_graph_states, reduction=True):
        z = suspect_graph_states
        ntl_loss = self.ntl_loss_fn(z, reduction=reduction)

        return {"NTL": ntl_loss}

    def get_contextual_OCC_NTL_loss(self, context_graph_state, suspect_graph_states, reduction=True):
        B, K, d = suspect_graph_states.shape
        ntl_loss = self.ntl_loss_fn(suspect_graph_states, reduction=reduction)

        contrast_loss = self.occ_loss_fn(context_graph_state.unsqueeze(1).repeat(1, K-1, 1), suspect_graph_states[:,1:], reduction=reduction)
        return {"NTL": ntl_loss, "OCC": contrast_loss}


    def get_contextual_OCC_NTL_Graph_loss(self, context_graph_state, suspect_graph_states, reduction=True):
        B, K, N, d = suspect_graph_states.shape
        ntl_loss = self.ntl_loss_fn(suspect_graph_states.sum(axis=-2), reduction=reduction)

        if reduction:
            feature_contrast_loss = torch.mean(torch.mean(torch.mean(torch.norm(context_graph_state.unsqueeze(1).repeat(1, K-1, 1, 1)-suspect_graph_states[:,1:], p=2, dim=[-1]), axis=-1), axis=1))
            suspect_graphs = []
            for k in range(K-1):
                suspect_graphs.append(construct_cosine_graph(suspect_graph_states[:, k+1, :], topK=self.topK).unsqueeze(1))
            context_graph = construct_cosine_graph(context_graph_state, topK=self.topK)
            x2 = context_graph.unsqueeze(1).repeat(1, K-1, 1, 1)
            x1 = torch.cat(suspect_graphs, axis=1)
            graph_contrast_loss = torch.mean(torch.mean(torch.norm(x2 - x1, p="fro", dim=[-1, -2], keepdim=False)/(x2.shape[-1]**2), axis=1))
            return ntl_loss + feature_contrast_loss + graph_contrast_loss
        else:
            feature_contrast_loss = torch.mean(torch.mean(torch.norm(context_graph_state.unsqueeze(1).repeat(1, K-1, 1, 1)-suspect_graph_states[:,1:], p=2, dim=[-1]), axis=-1), axis=1)
            graph_contrast_loss = torch.mean(torch.norm(context_graph_state.unsqueeze(1).repeat(1, K-1, 1, 1)-suspect_graph_states[:,1:], p="fro", dim=[-1, -2], keepdim=False)/(context_graph_state.shape[-1]**2), axis=-1)
            return {"NTL": ntl_loss, "OCC": feature_contrast_loss, "Graph": graph_contrast_loss}

    def get_contextual_NTL_Graph_loss(self, context_graph_state, suspect_graph_states, reduction=True):
        B, K, N, d = suspect_graph_states.shape
        ntl_loss = self.ntl_loss_fn(suspect_graph_states.sum(axis=-2), reduction=reduction)

        if reduction:
            suspect_graphs = []
            for k in range(K-1):
                suspect_graphs.append(construct_cosine_graph(suspect_graph_states[:, k+1, :], topK=self.topK).unsqueeze(1))
            context_graph = construct_cosine_graph(context_graph_state, topK=self.topK)
            x2 = context_graph.unsqueeze(1).repeat(1, K-1, 1, 1)
            x1 = torch.cat(suspect_graphs, axis=1)
            graph_contrast_loss = torch.mean(torch.mean(torch.norm(x2 - x1, p="fro", dim=[-1, -2], keepdim=False)/(x2.shape[-1]**2), axis=1))
            return ntl_loss + graph_contrast_loss
        else:
            graph_contrast_loss = torch.mean(torch.norm(context_graph_state.unsqueeze(1).repeat(1, K-1, 1, 1)-suspect_graph_states[:,1:], p="fro", dim=[-1, -2], keepdim=False)/(context_graph_state.shape[-1]**2), axis=-1)
            return {"NTL": ntl_loss, "Graph": graph_contrast_loss}

    def get_loss(self, context_graph_state, suspect_graph_state, suspect_graph_states, reduction=True):

        if "contextual" in self.loss_type and "NTL" in self.loss_type and "OCC" in self.loss_type:
            return self.get_contextual_OCC_NTL_loss(context_graph_state, suspect_graph_states, reduction=reduction)
        elif "contextual" in self.loss_type and "OCC" in self.loss_type and "NTL" not in self.loss_type:
            return self.get_contextual_OCC_loss(context_graph_state, suspect_graph_state, reduction=reduction)
        elif "contextual" in self.loss_type and "infoNCE" in self.loss_type:
            return self.get_infoNCE_loss(context_graph_state, suspect_graph_state, reduction=reduction)
        elif "NTL" in self.loss_type:
            return self.get_NTL_loss(suspect_graph_states, reduction=reduction)