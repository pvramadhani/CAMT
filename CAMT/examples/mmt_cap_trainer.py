"""Standalone patched MMT trainer; original mmt/trainers.py untouched."""
import time
import torch
from torch.nn import functional as F
from mmt.evaluation_metrics import accuracy
from mmt.loss import CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from mmt.utils.meters import AverageMeter

class MMTProxyContrastTrainer(object):
    def __init__(self, model_1, model_2,
                       model_1_ema, model_2_ema, num_cluster=500, alpha=0.999, cap_memory=None, cap_weight=0.1):
        super(MMTProxyContrastTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha
        self.cap_memory = cap_memory
        self.cap_weight = cap_weight

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]
        losses_cap = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets, cams = self._parse_data(target_inputs)

            # forward
            f_out_t1, p_out_t1, bn_t1 = self.model_1(inputs_1, return_bn=True)
            f_out_t2, p_out_t2, bn_t2 = self.model_2(inputs_2, return_bn=True)
            p_out_t1 = p_out_t1[:,:self.num_cluster]
            p_out_t2 = p_out_t2[:,:self.num_cluster]

            f_out_t1_ema, p_out_t1_ema, bn_ema1 = self.model_1_ema(inputs_1, return_bn=True)
            f_out_t2_ema, p_out_t2_ema, bn_ema2 = self.model_2_ema(inputs_2, return_bn=True)
            p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                     (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            # Add ONE published cluster-memory contrastive term; preserve
            # original hard CE/triplet and mutual teacher soft losses exactly.
            off1, on1 = self.cap_memory(bn_t1, targets, cams)
            off2, on2 = self.cap_memory(bn_t2, targets, cams)
            loss_cap_1 = (off1 + on1) / 2
            loss_cap_2 = (off2 + on2) / 2
            loss = loss + self.cap_weight * (loss_cap_1 + loss_cap_2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update dictionary only AFTER backward to avoid version conflicts.
            with torch.no_grad():
                teacher_bn = F.normalize(bn_ema1.detach() + bn_ema2.detach(), dim=1)
                self.cap_memory.update(teacher_bn, targets, cams)

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            losses_cap.update(((loss_cap_1 + loss_cap_2)/2).item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_CAP {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              losses_cap.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids, cameras = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets, cameras.cuda()

    def probe(self, loader, ce_soft_weight=0.5):
        """Check CC gradient on BN weight vs original hard CE gradient."""
        self.model_1.train()
        inputs_1, inputs_2, targets, cams = self._parse_data(loader.next())
        f_out_t1, p_out_t1, bn = self.model_1(inputs_1, return_bn=True)
        p_out_t1 = p_out_t1[:, :self.num_cluster]
        # Offline and online proxy association against one mini-batch.
        off, on = self.cap_memory(bn, targets, cams)
        cc = (off + on) / 2
        ce = self.criterion_ce(p_out_t1, targets)
        bn_weight = self.model_1.module.feat_bn.weight
        g_cc = torch.autograd.grad(self.cap_weight * cc, bn_weight,
                                   retain_graph=True, allow_unused=False)[0]
        g_ce = torch.autograd.grad((1-ce_soft_weight) * ce, bn_weight,
                                   allow_unused=False)[0]
        num = g_cc.norm().item()
        den = g_ce.norm().item()
        ratio = num / max(den, 1e-12)
        print('CAP PROBE: n_clusters={} n_proxies={} batch_unique={} off_loss={:.4f} on_loss={:.4f} cap_loss={:.4f} ce_loss={:.4f} '
              'cc_grad_norm={:.6g} weighted_ce_grad_norm={:.6g} grad_ratio={:.4f}'.format(
              self.num_cluster, self.cap_memory.num_proxies, targets.unique().numel(),
              off.item(), on.item(), cc.item(), ce.item(),
              num, den, ratio))
        if not (torch.isfinite(cc).item() and 0.01 <= ratio <= 2.0):
            print('CAP PROBE WARNING: gradient ratio outside exploratory 0.01..2.0 band; '
                  'do not start full training before inspecting.')
