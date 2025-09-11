import torch
from copy import deepcopy
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from tqdm import tqdm #added
from .incremental_learning import Inc_Learning_Appr_PPNet
from networks.tesnet import TesNet
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr_PPNet):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, use_pipnet = False, pipnet_loss = None, nepochs=100, lr=0.05, lr_min=1e-4,
                 lr_factor=3, lr_patience=5, clipgrad=10000, momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, 
                 wu_lr_factor=1, fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1,
                 T=2, perc=5, similarity_reg=False, normalize_sim=False,
                 lr_old=None, permute_settlement=False):
        super(Appr, self).__init__(model, device, use_pipnet, pipnet_loss, nepochs, lr, lr_min, lr_factor,
                                   lr_patience, clipgrad, momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor,
                                   fix_bn, eval_on_train, logger, exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.perc = perc
        self.similarity_reg = similarity_reg
        self.normalize_sim = normalize_sim
        self.settlers = None
        self.permute_settlement = permute_settlement
        if lr_old:
            self.lr_old = lr_old
        else:
            self.lr_old = 3 * self.lr

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=0.25, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--perc', default=97, type=int, required=False,
                            help='Percentile to mask the acts')
        parser.add_argument('--similarity_reg', action='store_true',
                            help='Whether to use similarity or distances to regularize')
        parser.add_argument('--normalize_sim', action='store_true',
                            help='Whether to normalize the similarities')
        parser.add_argument('--lr_old', default=0.25, type=float, required=False,
                            help='Learning rate')
        parser.add_argument('--permute_settlement', action='store_true',
                            help='Whether to permute protots to settlement')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _get_optimizers(self, t):
        """Returns the optimizer"""
        if not self.model.model.share_add_ons:
            warm_params = [
                {'params': self.model.heads[t].add_on_layers.parameters(), 'lr': 3 * self.lr, 'weight_decay': self.wd},
                {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
            ]
            joint_params = [
                {'params': self.model.model.features.parameters(), 'lr': self.lr / 10, 'weight_decay': self.wd},
                {'params': self.model.heads[t].add_on_layers.parameters(), 'lr': 3 * self.lr,
                 'weight_decay': self.wd},
                {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
            ]
            push_params = [{'params': self.model.heads[t].last_layer.parameters(), 'lr': self.lr,
                            'weight_decay': self.wd},
                           ]
        else:
            warm_params = [
                {'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr, 'weight_decay': self.wd},
                {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
            ]
            joint_params = [
                {'params': self.model.model.features.parameters(), 'lr': self.lr / 10, 'weight_decay': self.wd},
                {'params': self.model.model.add_on_layers.parameters(), 'lr': 3 * self.lr,
                 'weight_decay': self.wd},
                {'params': self.model.heads[t].prototype_vectors, 'lr': 3 * self.lr},
            ]
            push_params = [{'params': self.model.heads[t].last_layer.parameters(), 'lr': self.lr,
                            'weight_decay': self.wd},
                           ]
        if t > 0:
            joint_params.extend([
                {'params': self.model.heads[i].prototype_vectors, 'lr': self.lr_old} for i in range(t)
            ])
            warm_params.extend([
                {'params': self.model.heads[i].prototype_vectors, 'lr': self.lr_old} for i in range(t)
            ])
            if not self.model.model.share_add_ons:
                joint_params.extend([
                    {'params': self.model.heads[i].add_on_layers.parameters(), 'lr': self.lr_old} for i in range(t)
                ])
                warm_params.extend([
                    {'params': self.model.heads[i].add_on_layers.parameters(), 'lr': self.lr_old} for i in range(t)
                ])


        warm_optimizer = torch.optim.Adam(warm_params)
        joint_optimizer = torch.optim.Adam(joint_params)
        proto_optimizer = torch.optim.Adam(push_params)
        return joint_optimizer, proto_optimizer, warm_optimizer

    def train_loop(self, t, trn_loader, val_loader, push_loader=None, pipnet=False, epochs_pretrain=1, 
                   freeze_epochs=1,trainloader_pretraining=None, params_to_freeze=None, params_to_train=None, params_backbone=None,
                   optimizer_net=None, optimizer_classifier=None, scheduler_net=None, scheduler_classifier=None, device='cpu'):
        """Contains the epochs loop"""
        print("TRAIN LOOP in icicle")
        # # add exemplars to train_loader
        # if len(self.exemplars_dataset) > 0 and t > 0:
        #     trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
        #                                              batch_size=trn_loader.batch_size,
        #                                              shuffle=True,
        #                                              num_workers=trn_loader.num_workers,
        #                                              pin_memory=trn_loader.pin_memory)

        # if t > 0:
        #     self.settlement(push_loader, task=t)
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader,
                           val_loader,
                           push_loader,
                           pipnet=pipnet, 
                           epochs_pretrain=epochs_pretrain,
                           freeze_epochs=freeze_epochs,
                           trainloader_pretraining=trainloader_pretraining,
                           params_to_freeze=params_to_freeze, 
                           params_to_train=params_to_train,
                           params_backbone=params_backbone,
                           optimizer_net=optimizer_net,
                           optimizer_classifier=optimizer_classifier,
                           scheduler_net=scheduler_net,
                           scheduler_classifier=scheduler_classifier,
                           device=device) #VEDI TRAINLOOP IN class Inc_Learning_Appr_PPNet(Inc_Learning_Appr)
        

        # EXEMPLAR MANAGEMENT -- select training subset
        # self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
    
    def train_pipnet(self, t, train_loader, optimizer_net,
                     optimizer_classifier, scheduler_net, 
                     scheduler_classifier, epoch, device,
                     pretrain=False, finetune=False,
                     progress_prefix: str = 'Train Epoch',
                     epochs_pretrain=1):
        print("train pipnet")

        # Make sure the model is in train mode
        self.model.train()

        if pretrain:
            # Disable training of classification layer
            self.model.module._classification.requires_grad = False
            progress_prefix = 'Pretrain Epoch'
        else:
            # Enable training of classification layer (disabled in case of pretraining)
            self.model.module._classification.requires_grad = True

        # Store info about the procedure
        train_info = dict()
        total_loss = 0.
        total_acc = 0.

        iters = len(train_loader)
        # Show progress on progress bar. 
        train_iter = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=progress_prefix+'%s'%epoch,
                        mininterval=2.,
                        ncols=0)
        len_train_iter = len(train_iter)
        count_param=0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                count_param+=1           
        print("Number of parameters that require gradient: ", count_param, flush=True)

        if pretrain:
            align_pf_weight = (epoch/epochs_pretrain)*1.
            unif_weight = 0.5 #ignored
            t_weight = 5.
            cl_weight = 0.
        else:
            align_pf_weight = 5. 
            t_weight = 2.
            unif_weight = 0.
            cl_weight = 2.


        print("Align weight: ", align_pf_weight,
              ", U_tanh weight: ", t_weight,
              "Class weight:", cl_weight, flush=True)
        print("Pretrain?", pretrain,
              "Finetune?", finetune, flush=True)

        lrs_net = []
        lrs_class = []
        # Iterate through the data set to update leaves, prototypes and network
        # if pretrain==False:
            # for x in train_iter:
            #     print("TRAIN ITER[0]",x[0])
            #     print("TRAIN ITER[1]",x[1])
            #     assert 33 == 22
        for i, (x) in train_iter:   
            # if i==20:
            #     break
            xs1, xs2, ys = None, None, None
            if pretrain:
                xs1, xs2 ,ys = x[0].to(device), x[1].to(device), x[2].to(device)
            else:
                xs1, xs2, ys = x[0][0].to(device), x[0][1].to(device), x[1].to(device)
                
            # Reset the gradients
            optimizer_classifier.zero_grad(set_to_none=True)
            optimizer_net.zero_grad(set_to_none=True)

            # Perform a forward pass through the network
            proto_features, pooled, out = self.model(torch.cat([xs1, xs2]))
            loss_d = torch.zeros(1).to(device)
            loss_cw = torch.zeros(1).to(device)

            ####TRYING TO IMPLEMENT THE KNOWLEDGE REGULARIZATION
            if t > 0:
                proto_features_old, pooled_old, _ = self.model_old(torch.cat([xs1, xs2]))
                indeces = pooled_old.detach() >= 0.5
                #CONTROLLA COME FARE QUESTO PASSAGGIO 
                #DATO CHE BISOGNA IMPOSTARLO PER PIPNET
                # loss_d, _ = self.knowledge_distillation(t, proto_features, proto_features_old, indeces)
                # PROTOTYPE DISTILLATION
                # loss_d, _ = self.prototype_distillation(proto_features, proto_features_old, indeces)
                # print("WEIGHTS 2???",self.model.module._classification.weight.data.shape,flush=True)
                loss_cw = self.cosineweight_loss(self.model.module._classification.weight.data,
                                                     self.model_old.module._classification.weight.data,
                                                     pooled.detach())
                
            # print("ys type prima di loss",type(ys))
            loss_p, acc = self.calculate_loss_pipnet(t,
                                                    epoch,
                                                    i,
                                                    proto_features,
                                                    pooled,
                                                    out,
                                                    ys,
                                                    align_pf_weight,
                                                    t_weight,
                                                    unif_weight,
                                                    cl_weight,
                                                    self.model.module._classification.normalization_multiplier,
                                                    pretrain,
                                                    finetune,
                                                    train_iter,
                                                    verbose=True,
                                                    EPS=1e-8,
                                                    loss_d=loss_d.item(),
                                                    loss_cw=loss_cw.item())
            loss = loss_p + loss_d + loss_cw

            # Compute the gradient
            loss.backward()

            if not pretrain:
                optimizer_classifier.step()   
                scheduler_classifier.step(epoch - 1  + (i/iters))
                lrs_class.append(scheduler_classifier.get_last_lr()[0])

            if not finetune:
                optimizer_net.step()
                scheduler_net.step() 
                lrs_net.append(scheduler_net.get_last_lr()[0])
            else:
                lrs_net.append(0.)

            with torch.no_grad():
                total_acc+=acc
                total_loss+=loss.item()
                self.logger.log_scalar(task= -1 if pretrain else t, iter=i, name="loss_p",value=loss_p.item(),group='pretrain' if pretrain else 'train')
                self.logger.log_scalar(task= -1 if pretrain else t, iter=i, name="loss_d",value=loss_d.item(),group='pretrain' if pretrain else 'train')
                self.logger.log_scalar(task= -1 if pretrain else t, iter=i, name="loss_cw",value=loss_d.item(),group='pretrain' if pretrain else 'train')
                self.logger.log_scalar(task= -1 if pretrain else t, iter=i,name="loss",value=loss.item(),group='pretrain' if pretrain else 'train')

            # if not pretrain:
            #     with torch.no_grad():
            #         self.model.module._classification.weight.copy_(torch.clamp(self.model.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
            #         self.model.module._classification.normalization_multiplier.copy_(torch.clamp(self.model.module._classification.normalization_multiplier.data, min=1.0)) 
            #         if self.model.module._classification.bias is not None:
            #             self.model.module._classification.bias.copy_(torch.clamp(self.model.module._classification.bias.data, min=0.))  
            
            # if (epoch==1 or epoch==self.nepochs-1) and (i==0 or i==len_train_iter-1):
            #     print("TASK:",t, "epoca:",epoch, "iterazione:",i)
            #     # print("proto_features", proto_features.shape, flush=True)
            #     print("pooled",pooled.shape,flush=True)
            #     torch.set_printoptions(profile="full")
            #     # print(proto_features[0],flush=True)
            #     print("pooled",pooled,flush=True)
            #     torch.set_printoptions(profile="default")
                # print("TASK:",t,flush=True)
                # torch.set_printoptions(profile="full")
                # print("out",out,flush=True)
                # torch.set_printoptions(profile="default")
                            
            
        train_info['train_accuracy'] = total_acc/float(i+1)
        train_info['loss'] = total_loss/float(i+1)
        train_info['lrs_net'] = lrs_net
        train_info['lrs_class'] = lrs_class

        return train_info        
    
    def settlement(self, loader, task=0):
        with torch.no_grad():
            if task > 0:
                vals = []
                labels = []
                ####MODIFICARE STA PARTE PER FARE Proximity-based prototype initialization.(??)
                for i, (x) in enumerate(loader):
                    xs1, xs2, ys = None, None, None
                    xs1, xs2, ys = x[0][0].to(device), x[0][1].to(device), x[1].to(device)
                    proto_features, pooled, out = self.model(torch.cat([xs1, xs2]))
                    
                    
                    
                for it, (input_data, label) in enumerate(loader):
                    outputs = self.model(input_data.cuda())
                    distances = torch.cat([outputs[i][0] for i in range(len(outputs))], dim=1)
                    vals.append(distances.reshape(distances.shape[0], distances.shape[1], -1).cpu())
                    labels.append(label)
                vals = torch.cat(vals, dim=0)

                if isinstance(self.model.model, TesNet):
                    class_vals_flt = torch.unique(vals.flatten())[:50000]
                else:
                    class_vals_flt = torch.unique(vals.flatten())[::2**(task-1)]
                q_h = torch.quantile(class_vals_flt, 0.90)
                class_vals_flt = class_vals_flt[class_vals_flt < q_h]
                q_l = class_vals_flt.min()
                qs = (q_l, q_h)

                reps = []
                dists_all = []
                for it, (input_data, label) in enumerate(loader):
                    dists = []
                    for t_inner in range(task):
                        c, d = self.model.push_forward(input_data.cuda(), t=t_inner)
                        conv_fts = c
                        dists.append(d.cpu())
                    distances = torch.cat(dists, dim=1)
                    if isinstance(self.model.model, TesNet):
                        distances = distances
                    reps.append(conv_fts)
                    dists_all.append(distances)
                reps = torch.cat(reps, dim=0).cpu()
                dists_all = torch.cat(dists_all, dim=0).cpu()

                kmean = KMeans(n_clusters=self.model.heads[0].prototype_vectors.shape[0], max_iter=10)
                inclass_reps = reps.permute(0, 2, 3, 1).reshape(-1, reps.shape[1]).numpy()
                cond = ((dists_all.mean(1).flatten() <= qs[1]))  # * (dists_all.mean(1).flatten() >= qs[0]))
                set_of_reps = inclass_reps[cond]
                kmean.fit(set_of_reps)
                settlers = torch.tensor(kmean.cluster_centers_, 
                                        dtype=torch.float32).unsqueeze(2).unsqueeze(2).cuda()
                self.model.heads[-1].prototype_vectors.data.copy_(settlers)
                self.settlers = settlers

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # print("NESSUN post train process in icicle")
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        # self.model_old.freeze_all()

        for parameter in self.model_old.module.parameters():
            parameter.requires_grad = False

###########QUESTO EVAL NON VIENE USATO CON PIPNET
    # def eval(self, t, val_loader, device='cpu'):
    #     """Contains the evaluation code"""
    #     print("USING EVAL IN ICICLE.PY")
    #     test_iter = tqdm(enumerate(val_loader),
    #                     total=len(val_loader),
    #                     desc='Eval',
    #                     mininterval=5.,
    #                     ncols=0)
    #     with torch.no_grad():
    #         total_loss, total_acc_taw, total_acc_tag, total_num, total_clst, total_sep, total_l1, total_avg_sep, total_entropy = \
    #             0, 0, 0, 0, 0, 0, 0, 0, 0
    #         self.model.eval()
    #         for i, (x) in test_iter:
    #             xs1, xs2, ys = None, None, None
    #             xs1, xs2, ys = x[0][0].to(device), x[0][1].to(device), x[1].to(device)
    #             proto_features, pooled, out = self.model(torch.cat([xs1, xs2]))   
    #             if t > 0:
    #                 with torch.no_grad():
    #                     proto_features_old, _, _ = self.model_old(torch.cat([xs1, xs2]))
                    
    #             loss, acc = self.calculate_loss_pipnet(proto_features,
    #                                                pooled, out,
    #                                                ys,
    #                                                align_pf_weight,
    #                                                t_weight,
    #                                                unif_weight,
    #                                                cl_weight,
    #             self.model.module._classification.normalization_multiplier,
    #                                                pretrain,
    #                                                finetune,
    #                                                train_iter,
    #                                                print=True,
    #                                                EPS=1e-8)
    #             print("USING kd IN EVAL (ICICLE.PY)")
    #             loss_d, _ = self.knowledge_distillation(t, proto_features, proto_features_old)
    #             loss += loss_d
                
        #         # Log
        #         clst_loss_val, sep_loss_val, l1_loss, avg_sep_cost, orth_loss, sub_loss = self.protopnet_looses(
        #             min_distances,
        #             targets.to(self.device),
        #             t,
        #             all_out=self.exemplars_dataset is not None,
        #         )
        #         loss = entropy_loss + clst_loss_val * 0.8 + sep_loss_val * self.model.model.sep_weight + 1e-4 * l1_loss + \
        #                1e-4 * orth_loss - 1e-7 * sub_loss
        #         hits_taw, hits_tag = self.calculate_metrics(logits, targets)
        #         # Log
        #         total_loss += loss.item() * len(targets)
        #         total_entropy += entropy_loss.item() * len(targets)
        #         total_clst += clst_loss_val.item()
        #         total_sep += sep_loss_val.item()
        #         total_avg_sep += avg_sep_cost.item()
        #         total_l1 += l1_loss.item()
        #         total_acc_taw += hits_taw.sum().item()
        #         total_acc_tag += hits_tag.sum().item()
        #         total_num += len(targets)
        #     ppnet_losses = {
        #         'clst': total_clst,
        #         'sep': total_sep,
        #         'avg_sep': total_avg_sep,
        #         'l1': total_l1,
        #         'entropy': total_entropy / total_num,
        #     }
        # return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, ppnet_losses
        # return loss
    
    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce
    
    def knowledge_distillation(self, t, distances=None, old_distances=None, indeces=None): #ADDED
        loss = 0
        t_in = t
        if t > 0:
            # if self.model.model.repeat_task_0:
            #     t_in = t - 1
            # print("self.normalize_sim", self.normalize_sim, flush=True)
            if self.normalize_sim:#FALSE
                old_distances = old_distances / ((torch.sum(old_distances, dim=2).sum(2))[:, :, None, None] + 0.0001)
                distances = distances / (torch.sum(distances, dim=2).sum(2)[:, :, None, None] + 0.0001)
            with torch.no_grad():
                q = torch.quantile(old_distances.reshape(
                    [distances.shape[0],
                     distances.shape[1], -1]),
                    self.perc / 100, dim=2)
                print("q SHAPE", q.shape, flush=True)
                # print("q", q, flush=True)
                # print("self.similarity_reg", self.similarity_reg, flush=True)   
                if self.similarity_reg:#it was true
                    mask = old_distances >= q[:, :, None, None]
                else:
                    mask = old_distances <= q[:, :, None, None]
                
                print("mask shape",mask.shape, flush=True)
                print(indeces.shape, flush=True)
                print("distances shape",distances[indeces].shape, flush=True)
            # Knowledge distillation loss for all previous tasks
            loss += (self.lamb) * ((old_distances[indeces] - distances[indeces]) * mask[indeces]).view(distances.shape[0], - 1).norm(2, dim=1).sum()
            # print("type(loss_d)",type(loss))
        return loss, t_in

    def prototype_distillation(self,distances=None, old_distances=None, indices=None): #ADDED
        loss = 0
        if self.normalize_sim:#FALSE
            old_distances = old_distances / ((torch.sum(old_distances, dim=2).sum(2))[:, :, None, None] + 0.0001)
            distances = distances / (torch.sum(distances, dim=2).sum(2)[:, :, None, None] + 0.0001)
        # Knowledge distillation loss for all previous tasks
        loss += (self.lamb) * ((old_distances[indices] - distances[indices])).view(distances[indices].shape[0], - 1).norm(2, dim=1).sum()
        # print("type(loss_d)",type(loss))
        return loss

    def cosineweight_loss(self, w, w_old, pooled):
        idx = torch.where(pooled > 0.5, pooled, torch.zeros_like(pooled)).sum(dim=0) > 0
        # print("IDX",idx)
        # print(idx.shape)
        lamb1 = 0.9
        loss_cwl = lamb1 * torch.nn.functional.cosine_similarity(w[:,idx], w_old[:,idx], dim=0).mean()
        return loss_cwl
    
    def criterion(self, t, outputs, targets, distances=None, old_distances=None):
        """Returns the loss value"""
        loss, t_in = self.knowledge_distillation(t, distances, old_distances) #i put the code in this function in incremental_learning.py
        # Current cross-entropy loss -- with exemplars use all heads
        if not self.use_pipnet and len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.module.task_offset[t_in]) # dato che net è in dataparallel con pipnet provo ad usare .module.task_offset
        # print("OUTPUTS[t]:",outputs[t].shape)
        # print("TARGETS:", targets.shape, targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets)
    

    