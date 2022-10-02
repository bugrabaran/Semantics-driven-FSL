import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.models import FewShotModel
import math

def euclidean_distance(feats, prototypes):
    # Convert the tensors in a shape that is easily computable
    n = feats.size(0)
    m = prototypes.size(0)
    d = feats.size(2)

    if d != prototypes.size(1):
        raise ValueError("Features and prototypes are of different size")

    prototypes = prototypes.unsqueeze(0).expand(n, m, d)
    return torch.pow(feats - prototypes, 2).sum(dim = 2)

class CombinedProtoNet(FewShotModel):
    def __init__(self, args, sem_feat_dim = 300):
        super().__init__(args)
        if args.backbone_class == "ConvNet":
            hdim = 64
        elif args.backbone_class == "Res12":
            hdim = 640
        elif args.backbone_class == "Res18":
            hdim = 512
        elif args.backbone_class == "WRN":
            hdim = 640
        else:
            raise ValueError("Unknown Backbone %s"%args.backbone_class)
        
        self.args = args
        self.sem_feat_dim = sem_feat_dim
        self.hidden_dim = hdim 
        self.encode_vis = nn.Sequential(nn.Linear(hdim, 32),
                                        nn.Dropout(p=0.2),
                                        nn.ReLU(),
                                        nn.Linear(32, 32))
        self.encode_sem = nn.Sequential(nn.Linear(sem_feat_dim, 32),
                                        nn.Dropout(p=0.6),
                                        nn.ReLU(),
                                        nn.Linear(32, 32))
        self.act = nn.ReLU()
        ###################################################################
        self.pred_hadamard = nn.Sequential(nn.Linear(sem_feat_dim, 32),
                                           nn.Softmax(dim=1),
                                           nn.Linear(32, hdim)
                                           )
        ##################################################################

        self.g_linear1 = nn.Linear(sem_feat_dim, sem_feat_dim)
        self.dropout_g = nn.Dropout(p=0.4)
        self.g_linear2 = nn.Linear(sem_feat_dim, hdim)
        
    def _forward(self, instance_embs, attrib, support_idx, query_idx):
        args = self.args
        emb_dim = instance_embs.size(-1)

        ########################################################################################
        e_attrib = attrib.repeat_interleave(args.shot, dim=0)


        # organize support/query data
        support = instance_embs.view(args.way, args.shot+args.query, emb_dim)[:, :args.shot, :].contiguous().view(args.way*args.shot, emb_dim)
        query   = instance_embs.view(args.way, args.shot+args.query, emb_dim)[:, args.shot:, :].contiguous().view(args.way*args.query, emb_dim)
                
        enc_support = self.encode_vis(support)
        enc_attrib = self.encode_sem(e_attrib)
                       
        # sample attention
        scale_inps = nn.Softmax(dim=1)(torch.bmm(enc_support.unsqueeze(dim=1), enc_attrib.unsqueeze(dim=2) / math.sqrt(32)).view(args.way, args.shot, 1))

        weighted_inp_feats = scale_inps.view(support.size(0), 1) * support
        class_feats = weighted_inp_feats.view(args.way, args.shot, emb_dim).sum(dim=1)

        hadamard = self.pred_hadamard(attrib)
        
        sem_proto = self.g_linear2(self.act(self.dropout_g(self.g_linear1(attrib))))
        
        prototypes =  args.vis_rate * (class_feats * hadamard) + (1 - args.vis_rate) * sem_proto 
       
        logits = -euclidean_distance(query.unsqueeze(1).expand(args.query*args.way, args.way, emb_dim) * hadamard, prototypes) / args.temperature
        
        if self.training:
            return logits, None
        else:
            return logits
