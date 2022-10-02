import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

def euclidean_distance(feats, prototypes):
    # Convert the tensors in a shape that is easily computable
    n = feats.size(0)
    m = prototypes.size(0)
    d = feats.size(2)

    if d != prototypes.size(1):
        raise ValueError("Features and prototypes are of different size")

    prototypes = prototypes.unsqueeze(0).expand(n, m, d)
    return torch.pow(feats - prototypes, 2).sum(dim = 2)

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward(self, instance_embs, support_idx, query_idx):
        args = self.args
        emb_dim = instance_embs.size(-1) 

        # organize support/query data
        support = instance_embs.view(args.way, args.shot+args.query, emb_dim)[:, :args.shot, :].contiguous().view(args.way*args.shot, emb_dim)
        query   = instance_embs.view(args.way, args.shot+args.query, emb_dim)[:, args.shot:, :].contiguous().view(args.way*args.query, emb_dim)

        prototypes = support.view(args.way, args.shot, emb_dim).mean(dim=1)

        logits = -euclidean_distance(query.unsqueeze(1).expand(args.query*args.way, args.shot, emb_dim), prototypes) / args.temperature

        if self.training:
            return logits, None
        else:
            return logits
