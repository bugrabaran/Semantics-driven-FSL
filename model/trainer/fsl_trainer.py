import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader, self.attributes = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler, self.optimizer_warmup, self.lr_warmup_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self, mode="train"):
        args = self.args

        # prepare one-hot label
        if mode == "train": 
            label = torch.arange(args.way, dtype=torch.int16).repeat_interleave(args.query)
        else:
            label = torch.arange(args.eval_way, dtype=torch.int16).repeat_interleave(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + args.warmup_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            batch_loss = 0
            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]
                
                #gt_label = gt_label.view(args.way, args.shot + args.query)[:, 1:]  for noisy cases
                gt_label = np.unique(gt_label.cpu().numpy()) 
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)
                # get saved centers
                logits, reg_logits = self.para_model(data, torch.from_numpy(self.attributes[gt_label]).cuda())
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                else:
                    loss = F.cross_entropy(logits, label) / args.num_acc
                    total_loss = F.cross_entropy(logits, label) / args.num_acc
                    
                tl2.add(loss)
                batch_loss += total_loss.item()
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)
                tl1.add(total_loss.item())
                ta.add(acc)
                if epoch < args.warmup_epoch + 1: 
                    if self.train_step % args.num_acc == 0:
                        self.optimizer_warmup.zero_grad()
                    total_loss.backward()
                    backward_tm = time.time()
                    self.bt.add(backward_tm - forward_tm)
                    if self.train_step % args.num_acc == 0:
                        self.optimizer_warmup.step()
                    optimizer_tm = time.time()
                    self.ot.add(optimizer_tm - backward_tm) 
                else:
                    if self.train_step % args.num_acc == 0:
                        self.optimizer.zero_grad()
                    total_loss.backward()
                    backward_tm = time.time()
                    self.bt.add(backward_tm - forward_tm)
                    if self.train_step % args.num_acc == 0:
                        self.optimizer.step()
                    optimizer_tm = time.time()
                    self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()
            
            if epoch > args.warmup_epoch + 1:
                self.lr_scheduler.step()
            else:
                self.lr_warmup_scheduler.step()
            print('Average loss over training batches %.4f'%(batch_loss / args.episodes_per_epoch))
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label, _ = self.prepare_label(mode="val")
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, r_lbl = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                if args.dataset == "TieredImageNet":
                    r_lbl = r_lbl + 351
                else:
                    r_lbl = r_lbl + 64 
                
                r_lbl = r_lbl.view(args.way, args.shot + args.query)[:, 1:]
                r_lbl = np.unique(r_lbl.cpu().numpy())
                logits = self.model(data, torch.from_numpy(self.attributes[r_lbl]).cuda())
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])       
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model.eval()
        record = np.zeros((10000, 2)) # loss and acc
        label, _ = self.prepare_label("test")
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, r_lbl = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
               
                if args.dataset == "TieredImageNet":
                    r_lbl = r_lbl + 448
                else:
                    r_lbl = r_lbl + 80
                 
                r_lbl = r_lbl.view(args.way, args.shot + args.query)[:, 1:]
                r_lbl = np.unique(r_lbl.cpu().numpy())
                start = time.time()
                logits = self.model(data, torch.from_numpy(self.attributes[r_lbl]).cuda())
                end = time.time()
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            


