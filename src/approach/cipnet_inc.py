import time
import torch
from copy import deepcopy
from argparse import ArgumentParser
from tqdm import tqdm #added
from .incremental_learning import Inc_Learning_Appr_PPNet
from datasets.exemplars_dataset import ExemplarsDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from networks.cipnet_utils.test import eval_cipnet
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class Appr(Inc_Learning_Appr_PPNet):
    """Class for the incremental learning approach for CIPNet
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, network = 'convnext', cipnet_loss = None, use_hoyer=False, use_proto_reg=False,
                 proto_reg_CE=False, nepochs=100, lr=0.05, lr_net=0.0005, lr_block=0.0005, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000, momentum=0,
                 wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False, logger=None,
                 exemplars_dataset=None, lamb=1, T=2, perc=5, similarity_reg=False, normalize_sim=False, lr_old=None,
                 permute_settlement=False):
        super(Appr, self).__init__(model, device, network, cipnet_loss, use_hoyer, use_proto_reg, proto_reg_CE, nepochs, lr, lr_min, lr_factor,
                                   lr_patience, clipgrad, momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor,
                                   fix_bn, eval_on_train, logger, exemplars_dataset)
        self.network = network
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.perc = perc
        self.similarity_reg = similarity_reg
        self.normalize_sim = normalize_sim
        self.settlers = None
        self.permute_settlement = permute_settlement
        self.lr = lr
        self.lr_net = lr_net
        self.lr_block = lr_block
        self.lamb_proto = 0.001
        self.lamb_hoyer = 100.0
        if lr_old:
            self.lr_old = lr_old
        else:
            self.lr_old = 3 * self.lr
        
        self.results_path = ""
        self.exp_name = ""
        self.num_tasks = 4
        self.classes_pertask = 50
        self.datasetname = "cub_200_2011_cropped_cipnet_inc_"
        
        
        # self.proto_used = {tt:{k:0 for k in range(self.model.module._num_prototypes)} for tt in range(4)}
        self.proto_used = {k:torch.zeros(768) for k in range(self.num_tasks)}
        self.proto_idxs = torch.zeros_like(self.proto_used[0],dtype=torch.bool)
        print("NUM PROTO IN INCLEARN", self.model.module._num_prototypes, type(self.model.module._num_prototypes), flush=True)


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

    def get_optimizer_nn(self) -> torch.optim.Optimizer:

        #create parameter groups
        params_to_freeze = []
        params_to_train = []
        params_backbone = []
        # set up optimizer
        if 'resnet50' in self.network: 
            # freeze resnet50 except last convolutional layer
            for name,param in self.model.module._net.named_parameters():
                if 'layer4.2' in name:
                    params_to_train.append(param)
                elif 'layer4' in name or 'layer3' in name:
                    params_to_freeze.append(param)
                elif 'layer2' in name:
                    params_backbone.append(param)
                else: #such that model training fits on one gpu. 
                    param.requires_grad = False
                    params_backbone.append(param)###vedi
        
        elif 'convnext' in self.network:
            print("chosen network is convnext", flush=True)
            for name,param in self.model.module._net.named_parameters():
                if 'features.7.2' in name: 
                    params_to_train.append(param)
                elif 'features.7' in name or 'features.6' in name:
                    params_to_freeze.append(param)
                # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
                # elif 'features.5' in name or 'features.4' in name:
                #     params_backbone.append(param)
                # else:
                #     param.requires_grad = False
                else:
                    params_backbone.append(param)
        else:
            print("Network is not ResNet or ConvNext.", flush=True)     
        classification_weight = []
        classification_bias = []
        for name, param in self.model.module._classification.named_parameters():
            if 'weight' in name:
                classification_weight.append(param)
            elif 'multiplier' in name:
                param.requires_grad = False
    #         else:
    #             if args.bias:
    #                 classification_bias.append(param)

        if hasattr(self.model.module, "tau"):
            paramlist_net = [
                    {"params": params_backbone, "lr": self.lr_net, "weight_decay_rate": self.wd},
                    {"params": params_to_freeze, "lr": self.lr_block, "weight_decay_rate": self.wd},
                    {"params": params_to_train, "lr": self.lr_block, "weight_decay_rate": self.wd},
                    {"params": self.model.module._add_on.parameters(), "lr": self.lr_block*10., "weight_decay_rate": self.wd},
                    {"params": [self.model.module.tau], "lr": self.lr_net*100.0, "weight_decay_rate": 0.0},]
            

        else:
            paramlist_net = [
                {"params": params_backbone, "lr": self.lr_net, "weight_decay_rate": self.wd},
                {"params": params_to_freeze, "lr": self.lr_block, "weight_decay_rate": self.wd},
                {"params": params_to_train, "lr": self.lr_block, "weight_decay_rate": self.wd},
                {"params": self.model.module._add_on.parameters(), "lr": self.lr_block*10., "weight_decay_rate": self.wd},]

        if hasattr(self.model.module._classification[-1], "tau"):
            paramlist_classifier = [
                    {"params": classification_weight, "lr": self.lr, "weight_decay_rate": self.wd},
                    {"params": classification_bias, "lr": self.lr, "weight_decay_rate": self.wd},
                    {"params": [self.model.module._classification[-1].tau], "lr": self.lr*10.0, "weight_decay_rate": 0.0},
            ]
        else:
            paramlist_classifier = [
                    {"params": classification_weight, "lr": self.lr, "weight_decay_rate": self.wd},
                    {"params": classification_bias, "lr": self.lr, "weight_decay_rate": self.wd},
            ]
            
        optimizer_net = torch.optim.AdamW(paramlist_net, lr=self.lr, weight_decay=self.wd)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier, lr=self.lr, weight_decay=self.wd)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone

    def get_optimizer_newhead(self):
        classification_weight = []
        classification_bias = []
        for name, param in self.model.module._classification[-1].named_parameters():
            if 'weight' in name:
                classification_weight.append(param)
            elif 'multiplier' in name:
                param.requires_grad = False
    #         else:
    #             if args.bias:
    #                 classification_bias.append(param)
        if hasattr(self.model.module._classification[-1], "tau"):
            paramlist_classifier = [
                    {"params": classification_weight, "lr": self.lr, "weight_decay_rate": self.wd},
                    {"params": classification_bias, "lr": self.lr, "weight_decay_rate": self.wd},
                    {"params": [self.model.module._classification[-1].tau], "lr": self.lr*10.0, "weight_decay_rate": 0.0},
            ]
        else:        
            paramlist_classifier = [
                    {"params": classification_weight, "lr": self.lr, "weight_decay_rate": self.wd},
                    {"params": classification_bias, "lr": self.lr, "weight_decay_rate": self.wd},
            ]
            
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier, lr=self.lr, weight_decay=self.wd)
        return optimizer_classifier

    def train(self, t, trn_loader, val_loader, push_loader=None, epochs_pretrain=0, freeze_epochs=10, params_to_freeze=None,
              params_to_train=None, params_backbone=None, optimizer_net=None, optimizer_classifier=None,
              scheduler_net=None, scheduler_classifier=None, device='cpu', seed = 1):
        """Main train structure"""
        if hasattr(self.model.module, 'tau'):
            self.model.module.tau.requires_grad_(True)
        elif hasattr(self.model.module._classification[t], 'tau'):
            self.model.module._classification[t].tau.requires_grad_(True)
        val_acc = self.train_loop(t, trn_loader, val_loader, push_loader=push_loader, epochs_pretrain=epochs_pretrain, 
                        freeze_epochs=freeze_epochs, params_to_freeze=params_to_freeze, params_to_train=params_to_train,
                        params_backbone=params_backbone, optimizer_net=optimizer_net, optimizer_classifier=optimizer_classifier,
                        scheduler_net=scheduler_net, scheduler_classifier=scheduler_classifier, device=device)
        self.post_train_process(t)
   
        total_elements = self.proto_used[t].numel()
        total_sum = self.proto_used[t].sum()
        total_sq_sum = (self.proto_used[t]**2).sum()

        global_mean = total_sum / total_elements
        global_mean_sq = total_sq_sum / total_elements
        global_var = global_mean_sq - global_mean ** 2
        global_std = global_var.sqrt()

        # Normalize and sum
        norm_t = (self.proto_used[t] - global_mean) / torch.clamp(global_std, min=1e-4)
        perc = torch.quantile(norm_t,0.75)
        
        self.proto_idxs += norm_t >= perc
        p = "./"+str(self.results_path)+"/"+str(self.datasetname)+str(self.exp_name)+"/figures"
        print(p, flush=True)
        if not os.path.exists(p): 
            p = "./"+str(self.results_path)+"/"+str(self.exp_name)+"/figures"
    
        plt.cla()
        plt.clf()
        plt.bar(range(len(self.proto_used[t])),self.proto_used[t])
        plt.savefig(p+"/proto_hist_task_"+str(t)+".png")
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        print("SAVE HIST of protos",flush=True)

        return val_acc


    def train_loop(self, t, trn_loader, val_loader, push_loader=None, epochs_pretrain=0, freeze_epochs=1, params_to_freeze=None,
                   params_to_train=None, params_backbone=None, optimizer_net=None, optimizer_classifier=None, scheduler_net=None,
                   scheduler_classifier=None, device='cpu'): #STA USANDO QUESTO
        """Contains the epochs loop"""
        print("TASK N.",t)
        # lr = self.lr
        # best_loss = np.inf
        # best_acc = 0.0
        # patience = self.lr_patience
        
        frozen = True
        lrs_net = []
        lrs_classifier = []
        
        # Loop epochs
        print("INIZIO TRAINING")
        nepochs = self.nepochs+1 #if self.use_cipnet else self.nepochs
        start_epoch = 1 #if self.use_cipnet else 0
        # train_acc = 0
        # train_loss = 0
        val_acc = []
        for e in range(start_epoch,nepochs):
            self.model.train()
            # Train
            
            epochs_to_finetune = 1 #during finetuning, only train classification layer and freeze rest.
                                    # usually done for a few epochs (at least 1, more depends on size of dataset)
            if e <= epochs_to_finetune and epochs_pretrain > 0:
                print("cipnet finetuning", flush=True)
                for param in self.model.module._add_on.parameters():
                    param.requires_grad = False
                for param in params_to_train:
                    param.requires_grad = False
                for param in params_to_freeze:
                    param.requires_grad = False
                for param in params_backbone:
                    param.requires_grad = False
                finetune = True
            else:
                print("cipnet training", flush=True) 
                finetune=False          
                if frozen:
                    # unfreeze backbone
                    if e>(freeze_epochs):
                        print("unfreezing backbone", flush=True)
                        for param in self.model.module._add_on.parameters():
                            param.requires_grad = True
                        for param in params_to_freeze:
                            param.requires_grad = True
                        for param in params_to_train:
                            param.requires_grad = True
                        for param in params_backbone:
                            param.requires_grad = True   
                        frozen = False
                    # freeze first layers of backbone, train rest
                    else:
                        print("freezing backbone", flush=True)
                        for param in params_to_freeze:
                            param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                        for param in self.model.module._add_on.parameters():
                            param.requires_grad = True
                        for param in params_to_train:
                            param.requires_grad = True
                        for param in params_backbone:
                            param.requires_grad = False
            
            
            print("\n Epoch", e, "frozen:", frozen, flush=True)            
            train_info = self.train_cipnet(t, trn_loader, optimizer_net, optimizer_classifier, scheduler_net,
                                        scheduler_classifier, e, device, pretrain=False, finetune=finetune)
            lrs_net+=train_info['lrs_net']
            lrs_classifier+=train_info['lrs_class']   
            
            # Evaluate model
            print("doing eval at epoch", e, flush=True)
            self.model.eval()
            eval_info = self.validation(t, val_loader, e, device, finetune=finetune)
            val_acc.append(eval_info['val_accuracy_task'])

                
        print(self.proto_used[t])
        return val_acc

    def pretrain_cipnet(self, t, epochs_pretrain, freeze_epochs, trn_loader, device,
                        save_model, load_backbone, results_path, exp_name):
        optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = self.get_optimizer_nn()            
        scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trn_loader)*epochs_pretrain, eta_min=self.lr_net/100., last_epoch=-1)
        lrs_pretrain_net = []
        self.pretraining_cipnet = True
        if hasattr(self.model.module, 'tau'):
            self.model.module.tau.requires_grad_(False)
        elif hasattr(self.model.module._classification[t], 'tau'):
            self.model.module._classification[t].tau.requires_grad_(False)
        # PRETRAINING PROTOTYPES PHASE
        for epoch in range(1, epochs_pretrain+1):
            for param in params_to_train:
                param.requires_grad = True
            for param in self.model.module._add_on.parameters():
                param.requires_grad = True
            for param in self.model.module._classification.parameters():
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = True # can be set to False when you want to freeze more layers
            for param in params_backbone:
                param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)

            train_info = self.train_cipnet(t, trn_loader, optimizer_net, optimizer_classifier, scheduler_net, None,
                                           epoch, device, pretrain=True, finetune=False, epochs_pretrain=epochs_pretrain)
            lrs_pretrain_net+=train_info['lrs_net']
        del optimizer_net
        del scheduler_net
        if hasattr(self.model.module, 'tau'):
            self.model.module.tau.requires_grad_(True)
        elif hasattr(self.model.module._classification[t], 'tau'):
            self.model.module._classification[t].tau.requires_grad_(True)

    def train_cipnet(self, t, train_loader, optimizer_net,
                     optimizer_classifier, scheduler_net, 
                     scheduler_classifier, epoch, device,
                     pretrain=False, finetune=False,
                     progress_prefix: str = 'Train Epoch',
                     epochs_pretrain=1):
        print("train cipnet")
        if self.model.parallelization=="DDP":
            train_loader.sampler.set_epoch(epoch)
        # Make sure the model is in train mode
        self.model.to(self.device)
        self.model.train()

        if pretrain:
            # Disable training of classification layer
            for j in range(len(self.model.module._classification)):
                for param in self.model.module._classification[j].parameters():
                    param.requires_grad = False
            progress_prefix = 'Pretrain Epoch'
            if hasattr(self.model.module, 'tau'):
                self.model.module.tau.requires_grad_(False)
            elif hasattr(self.model.module._classification[t], 'tau'):
                self.model.module._classification[t].tau.requires_grad_(False)
        else:
            # Enable training of classification layer (disabled in case of pretraining)
            for j in range(len(self.model.module._classification)):
                for param in self.model.module._classification[j].parameters():
                    if j == t:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            if hasattr(self.model.module, 'tau'):
                self.model.module.tau.requires_grad_(True)
            elif hasattr(self.model.module._classification[t], 'tau'):
                self.model.module._classification[t].tau.requires_grad_(True)
            

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
            t_weight = 1.0
            cl_weight = 0.
        else:
            align_pf_weight = 5. 
            t_weight = 1.0
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
        for i, (x) in train_iter:   
            xs1, xs2, ys = None, None, None
            if pretrain and self.pretraining_cipnet == False:
                xs1, xs2 ,ys = x[0].to(device), x[1].to(device), x[2].to(device)
            else:
                xs1, xs2, ys = x[0][0].to(device), x[0][1].to(device), x[1].to(device)
            if i == 10:
                break
            # Reset the gradients
            if not pretrain:
                optimizer_classifier[-1].zero_grad(set_to_none=True)

            else:
                optimizer_classifier.zero_grad(set_to_none=True)
            optimizer_net.zero_grad(set_to_none=True)

            # Perform a forward pass through the network
            proto_features, pooled, out = self.model(torch.cat([xs1, xs2]).to(device))

            #Keep count of which prototypes are used in the current task
            self.proto_used[t] += (pooled > 0.5).sum(dim=0).cpu()
                
            #obtain prototype features from old model
            proto_features_old = None
            if t > 0:
                self.model_old.eval()
                with torch.no_grad():
                    proto_features_old, _, _ = self.model_old(torch.cat([xs1, xs2]).to(device))

            loss, acc = self.calculate_loss_cipnet(t,
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
                                                    1.0,
                                                    pretrain=pretrain,
                                                    finetune=finetune,
                                                    train_iter=train_iter,
                                                    verbose=True,
                                                    EPS=1e-8,
                                                    old_proto=proto_features_old,
                                                    device=device)
            

            # Compute the gradient
            loss.backward()

            if not pretrain:
                optimizer_classifier[-1].step()   
                scheduler_classifier[-1].step(epoch - 1  + (i/iters))
                lrs_class.append(scheduler_classifier[-1].get_last_lr()[0])

            if not finetune:
                optimizer_net.step()
                scheduler_net.step() 
                lrs_net.append(scheduler_net.get_last_lr()[0])
            else:
                lrs_net.append(0.)

            with torch.no_grad():
                total_acc+=acc
                total_loss+=loss.item()
            
        train_info['train_accuracy'] = total_acc/float(i+1)
        train_info['train_iters'] = float(i+1)
        train_info['loss'] = total_loss/float(i+1)
        train_info['lrs_net'] = lrs_net
        train_info['lrs_class'] = lrs_class

        return train_info  


    def validation(self, t, val_loader, epoch, device, finetune=False,
                     progress_prefix: str = 'Val Epoch'):
        with torch.no_grad():
            val_info = dict()
            total_loss = 0.
            total_acc = 0.
            align_pf_weight = 5. 
            t_weight = 1.
            unif_weight = 0.
            cl_weight = 2.

            val_iter = tqdm(enumerate(val_loader),
                            total=len(val_loader),
                            desc=progress_prefix+' %s'%epoch,
                            mininterval=5.,
                            ncols=0)
            (xs, ys) = next(iter(val_loader))  
            for i, (xs, ys) in val_iter:
                if i == 10:
                    break
                xs, ys = xs.to(device), ys.to(device)
                proto_features, pooled, out = self.model(xs)
                
                hoyer_loss = torch.tensor(0.0).to(device)
                dec_loss = torch.tensor(0.0).to(device)
                
                proto_features_old = None
                if t > 0:
                    with torch.no_grad():
                        proto_features_old, _, _ = self.model_old(xs)

                if self.use_hoyer:
                    hoyer_loss = self.hoyer_loss()

                loss = torch.tensor(0.0).to(device)
                a_loss_pf = torch.tensor(0.0).to(device)
                loss_pr1 = torch.zeros(1).to(device)
                
                if proto_features_old is not None and self.use_proto_reg:
                        loss_pr1, inv_idxs = self.perhead_proto_reg(proto_features, proto_features_old, t=t, e=epoch)
                    
                        if inv_idxs.sum() == 0:
                            inv_idxs = torch.ones([self.model.module._num_prototypes]).bool()
            
                        tanh_loss = -torch.log(torch.tanh(torch.sum(pooled[:,inv_idxs],dim=0))+1e-8).mean() #L'_T IN PAPER
                    
                        loss += loss_pr1


                
                else:
                    tanh_loss = -torch.log(torch.tanh(torch.sum(pooled,dim=0))+1e-8).mean() #L_T IN PAPER

                ####################################################################
                if not finetune:
                    loss += align_pf_weight*a_loss_pf
                    loss += t_weight * tanh_loss


                log_softmax_inputs = F.log_softmax((out),dim=1)
                class_loss = self.cipnet_loss(log_softmax_inputs,ys) #L_C IN PAPER

                if finetune:
                    loss= cl_weight * class_loss
                else:
                    loss+= cl_weight * class_loss
                
                #Decorrelation loss
                dec_loss = self.heads_decorrelation(device)
                loss += dec_loss

                acc=0.
                start = t * self.classes_pertask
                end = (t + 1) * self.classes_pertask
                ys_pred_max_ag = torch.argmax(out, dim=1)
                ys_pred_max_aw = torch.argmax(out[:,start:end], dim=1)

                correct_ag = torch.sum(torch.eq(ys_pred_max_ag, ys))
                correct_aw = torch.sum(torch.eq(ys_pred_max_aw, ys%self.classes_pertask))
                
                acc_ag = correct_ag.item() / float(len(ys))
                acc_aw = correct_aw.item() / float(len(ys))
        
                if finetune:
                    if hasattr(self.model.module, "tau"):
                        val_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{hoyer_loss.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module.tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)
                    elif hasattr(self.model.module._classification[t], "tau"):
                        val_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{hoyer_loss.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module._classification[t].tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)
                    else:
                        val_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{hoyer_loss.item():.3f}, decL:{dec_loss.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)
                    self.logger.log_scalar(task= t, iter=i,name="LA",value=a_loss_pf.item(),group='val')
                    self.logger.log_scalar(task= t, iter=i,name="LT",value=tanh_loss.item(),group='val')
                else:
                    if hasattr(self.model.module, "tau"):
                        val_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{hoyer_loss.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module.tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)            
                    elif hasattr(self.model.module._classification[t], "tau"):
                        val_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{hoyer_loss.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module._classification[t].tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)            
                    else:
                        val_iter.set_postfix_str(
                        f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{hoyer_loss.item():.3f}, decL:{dec_loss.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)            
                    self.logger.log_scalar(task= t, iter=i,name="LA",value=(align_pf_weight*a_loss_pf).item(),group='val')
                    self.logger.log_scalar(task= t, iter=i,name="LT",value=(t_weight * tanh_loss).item(),group='val')
                
                self.logger.log_scalar(task=t, iter=i,name="LC",value=(cl_weight*class_loss).item(),group='val')
                total_acc+=acc_ag
                total_loss+=loss.item()
                # self.logger.log_scalar(task= -1 if pretrain else t, iter=i, name="loss_p",value=loss_p.item(),group='pretrain' if pretrain else 'train')
                self.logger.log_scalar(task= t, iter=i, name="loss_pr1",value=loss_pr1.item(),group='val')
                self.logger.log_scalar(task= t, iter=i, name="loss_ho",value=hoyer_loss.item(),group='val')
                self.logger.log_scalar(task= t, iter=i,name="loss",value=loss.item(),group='val')
                self.logger.log_scalar(task= t, iter=i, name="dec_loss",value=dec_loss.item(),group='val')

                val_info['val_accuracy_task'] = total_acc/float(i+1)
                val_info['val_loss'] = total_loss/float(i+1)
            
            return val_info
        
    def eval_test(self,task, u_task, tst_loader, device, path_results=None, path_exp=None, classes_pertask=50):
        tst_info = eval_cipnet(net=self.model,task=task, u_task=u_task, test_loader=tst_loader, epoch=0, device=device, progress_prefix='Test Epoch', classes_per_task=classes_pertask)
        self.logger.log_scalar(task= u_task, iter=0, name="num non-zero prototypes",value=tst_info['num non-zero prototypes'],group='test')
        self.logger.log_scalar(task= u_task, iter=0, name="test_accuracy_task",value=tst_info['test_accuracy_task'],group='test')
        self.logger.log_scalar(task= u_task, iter=0, name="test_accuracy_task_ag",value=tst_info['test_accuracy_task_ag'],group='test')
        self.logger.log_scalar(task= u_task, iter=0, name="almost_sim_nonzeros",value=tst_info['almost_sim_nonzeros'],group='test')
        self.logger.log_scalar(task= u_task, iter=0, name="local_size_all_classes",value=tst_info['local_size_all_classes'],group='test')
        self.logger.log_scalar(task= u_task, iter=0, name="almost_nonzeros",value=tst_info['almost_nonzeros'],group='test')
        self.logger.log_scalar(task= u_task, iter=0, name="top1_accuracy",value=tst_info['top1_accuracy'],group='test')
        self.logger.log_scalar(task= u_task, iter=0, name="top5_accuracy",value=tst_info['top5_accuracy'],group='test')
        
        print("TEST ACCURACY TASK", tst_info['test_accuracy_task'],flush=True)
        print("TEST ACCURACY TASK AGNOSTIC", tst_info['test_accuracy_task_ag'],flush=True)
        # print("CONFUSION MATRIX task", tst_info["confusion_matrix_task"].shape,flush=True)
        # # torch.set_printoptions(profile="full")
        # # print(tst_info["confusion_matrix_task"],flush=True)
        torch.set_printoptions(profile="default")
        f,ax = plt.subplots(1,1,figsize=(35,35))
        disp = ConfusionMatrixDisplay(confusion_matrix=tst_info["confusion_matrix_task"])
        disp.plot(ax=ax)
        p = "./"+str(path_results)+"/"+str(self.datasetname)+str(path_exp)+"/results/" 
        if not os.path.exists(p):
            p = "./"+str(path_results)+"/"+str(path_exp)+"/results/cm_train"+str(task)+"_test"+str(u_task)+".png"
        else:
            p = p+"cm_train"+str(task)+"_test"+str(u_task)+".png"
        # if self.model.rank == 0 or self.model.rank == -1:
        plt.savefig(p)
        plt.show()
        plt.clf()
        plt.close()

        print("CONFUSION MATRIX task", tst_info["confusion_matrix_task_ag"].shape,flush=True)
        # torch.set_printoptions(profile="full")
        # print(tst_info["confusion_matrix_task_ag"],flush=True)
        torch.set_printoptions(profile="default")
        f,ax = plt.subplots(1,1,figsize=(35,35))
        disp = ConfusionMatrixDisplay(confusion_matrix=tst_info["confusion_matrix_task_ag"])
        disp.plot(ax=ax, xticks_rotation=90)
        p = "./"+str(path_results)+"/"+str(self.datasetname)+str(path_exp)+"/results/" 
        if not os.path.exists(p):
            p = "./"+str(path_results)+"/"+str(path_exp)+"/results/cm_ag_train"+str(task)+"_test"+str(u_task)+".png"
        else:
            p = p+"cm_ag_train"+str(task)+"_test"+str(u_task)+".png"
        # if self.model.rank == 0 or self.model.rank == -1:
        plt.savefig(p)
        plt.show()
        plt.clf()
        plt.close()


        return tst_info

    def post_train_process(self, t):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        self.model.eval()
        for parameter in self.model.module._classification[-1].parameters():
            parameter.requires_grad = False
        self.model_old = deepcopy(self.model)
        self.model_old.eval()

        for parameter in self.model_old.module.parameters():
            parameter.requires_grad = False

    def perhead_proto_reg(self, proto_features, old_proto_features, indices=None, t=None, e=None):
        loss = 0.
        inv_idxs = torch.zeros_like(self.proto_used[0]).squeeze()
        for i in range(t):
            w = F.softplus(self.model.module._classification[i].weight) + 1e-6
            head_proto_loss, head_inv_idxs = self.prototype_regularization_loss(w,proto_features,old_proto_features,indices,i,e)
            loss += head_proto_loss
            inv_idxs += head_inv_idxs
        inv_idxs = torch.nonzero(inv_idxs, as_tuple=False)
        return loss, inv_idxs
    
    def prototype_regularization_loss(self, w_old, proto_features, old_proto_features, indices=None, t=None, e=None):
       
        idxs = torch.zeros_like(self.proto_used[0],dtype=torch.bool)
        
        total_elements = self.proto_used[t].numel()
        total_sum = self.proto_used[t].sum()
        total_sq_sum = (self.proto_used[t]**2).sum()

        global_mean = total_sum / total_elements
        global_mean_sq = total_sq_sum / total_elements
        global_var = global_mean_sq - global_mean ** 2
        global_std = global_var.sqrt()

        # Normalize and sum
        norm_t = (self.proto_used[t] - global_mean) / torch.clamp(global_std, min=1e-4)
        perc = torch.quantile(norm_t,0.75)    
        idxs += norm_t >= perc
    
        percent_true = idxs.float().mean().item() * 100
        inv_idxs = idxs<1
        idxs = torch.nonzero(idxs, as_tuple=False).squeeze()

        if percent_true > 80.0:
            print("Percent used prototipes:",percent_true, flush=True)
        
        diff = (old_proto_features[:,idxs,:,:] - proto_features[:,idxs,:,:])
        diff = diff.view(diff.size(0),diff.size(1), -1)
        norm_diff = diff.norm(2, dim=2)
        
        wmax = w_old.abs().max(dim=0).values.detach()
        weighted_norm = norm_diff * wmax[idxs]
        reg = self.lamb_proto
        loss = reg * weighted_norm.mean()
        
        return loss, inv_idxs
    
    def hoyer_loss(self, eps=1e-6):
        # Get the weight tensor from the final classification layer
        weights = self.model.module._classification[-1].weight  # shape: (out_features, in_features)
        
        # Flatten each weight vector
        weights_flat = weights.view(weights.size(0), -1)  # (num_vectors, vector_length)
        
        # Compute L1 and L2 norms per vector
        l1_norm = torch.norm(weights_flat, p=1, dim=1)
        l2_norm = torch.norm(weights_flat, p=2, dim=1) + eps  # add eps for stability
        
        # Number of elements in each vector
        n = weights_flat.size(1)
        
        # Compute Hoyer sparsity per vector
        hoyer_per_vector = (torch.sqrt(torch.tensor(float(n))) - (l1_norm / l2_norm)) / (torch.sqrt(torch.tensor(float(n))) - 1 + eps)
        
        # Average across all vectors
        hoyer = hoyer_per_vector.mean()
        
        # Define the loss (higher sparsity = lower loss)
        loss = self.lamb_hoyer * (1 - hoyer)
    
        return loss

    def heads_decorrelation(self, device): #L_D in paper
        dec_loss = torch.tensor(0.0).to(device)
        W_cur = F.normalize(F.softplus(self.model.module._classification[-1].weight) + 1e-6, dim=1)
        for i in range(len(self.model.module._classification)):
            W_pre = F.normalize(F.softplus(self.model.module._classification[i].weight) + 1e-6, dim=1)
            W = torch.matmul(W_pre, W_cur.T)**2
            dec_loss += (W).sum() - torch.diagonal(W).sum()
        return 0.005 * dec_loss 

    def calculate_loss_cipnet(self, t, epoch, iter, proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight,
                       cl_weight, net_normalization_multiplier, pretrain, finetune, train_iter, verbose=True, EPS=1e-10,
                       old_proto=None, device='cpu'):
        ys = torch.cat([ys1,ys1])
        pooled1, pooled2 = pooled.chunk(2)
        pf1, pf2 = proto_features.chunk(2)
        embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
        loss = torch.tensor(0.0).to(device)
        dec_loss = torch.tensor(0.0).to(device)

        with torch.no_grad():
            embv_avg = (embv1+embv2)/2.
        a_loss_pf = (align_loss(embv1, embv_avg)+ align_loss(embv2, embv_avg))/2. #ALIGNMENT LOSS L_A in paper

        loss_pr1 = torch.zeros(1).to(device)
        if old_proto is not None and self.use_proto_reg:
                #L_R in paper
                loss_pr1, inv_idxs = self.perhead_proto_reg(proto_features, old_proto, t=t, e=epoch) #L_R in paper
                if inv_idxs.sum() == 0:
                    inv_idxs = torch.ones([self.model.module._num_prototypes*2]).bool()
                inv_idxs1, inv_idxs2 = inv_idxs.chunk(2)
    
                tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1[:,inv_idxs1],dim=0))+EPS).mean() +\
                            torch.log(torch.tanh(torch.sum(pooled2[:,inv_idxs2],dim=0))+EPS).mean())/2 #L'_T in paper
            
                loss += loss_pr1
        else:
            #L_T in paper
            tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2. #L_T IN PAPER
        
        if not finetune:
            loss += align_pf_weight*a_loss_pf
            loss += t_weight * tanh_loss

        log_softmax_inputs = F.log_softmax((out),dim=1) 
        loss_ho = torch.tensor(0.0).to(device)
    
        if not pretrain:
            class_loss = self.cipnet_loss(log_softmax_inputs,ys) #L_C IN PAPER
            

            if self.use_hoyer:
                loss_ho = self.hoyer_loss() #L_H in paper
            
            dec_loss = self.heads_decorrelation(device) #L_D in paper
       
        loss += loss_ho
    
        if pretrain:
            class_loss = torch.tensor(0.0).to(device)
        loss += cl_weight * class_loss + dec_loss

        acc=0.
        start = t * self.classes_pertask
        end = (t + 1) * self.classes_pertask
        if not pretrain:
            ys_pred_max_ag = torch.argmax(out, dim=1)
            ys_pred_max_aw = torch.argmax(out[:,start:end], dim=1)
                
            correct_ag = torch.sum(torch.eq(ys_pred_max_ag, ys))
            correct_aw = torch.sum(torch.eq(ys_pred_max_aw, ys%self.classes_pertask))

            acc_ag = correct_ag.item() / float(len(ys))
            acc_aw = correct_aw.item() / float(len(ys))
        if verbose: 
            with torch.no_grad():
                if pretrain:
                    #print("PRETRAIN")
                    if hasattr(self.model.module, "tau"):
                        train_iter.set_postfix_str(
                        f'L: {loss.item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, tau:{self.model.module.tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
                    elif hasattr(self.model.module._classification[t], "tau"):
                        train_iter.set_postfix_str(
                        f'L: {loss.item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, tau:{self.model.module._classification[t].tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
                    else:
                        train_iter.set_postfix_str(
                        f'L: {loss.item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
                    self.logger.log_scalar(task= -1, iter=iter,name="LA",value=(align_pf_weight*a_loss_pf).item(),group='pretrain')
                    self.logger.log_scalar(task= -1, iter=iter,name="LT",value=(t_weight * tanh_loss).item(),group='pretrain')
                    self.logger.log_scalar(task= -1, iter=iter,name="LC",value=class_loss.item(),group='pretrain')
                else:
                    if finetune:
                        if hasattr(self.model.module, "tau"):
                            train_iter.set_postfix_str(
                            f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module.tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)
                        elif hasattr(self.model.module._classification[t], "tau"):
                            train_iter.set_postfix_str(
                            f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module._classification[t].tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)
                        else:
                            train_iter.set_postfix_str(
                            f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, decL:{dec_loss.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)
                        self.logger.log_scalar(task= t, iter=iter,name="LA",value=a_loss_pf.item(),group='train')
                        self.logger.log_scalar(task= t, iter=iter,name="LT",value=tanh_loss.item(),group='train')
                    else:
                        if hasattr(self.model.module, "tau"):
                            train_iter.set_postfix_str(
                            f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module.tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)            
                        elif hasattr(self.model.module._classification[t], "tau"):
                            train_iter.set_postfix_str(
                            f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, decL:{dec_loss.item():.2f}, tau:{self.model.module._classification[t].tau.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)            
                        else:
                            train_iter.set_postfix_str(
                            f'L:{loss.item():.3f},LC:{(cl_weight * class_loss).item():.3f}, LA:{(align_pf_weight*a_loss_pf).item():.2f}, LT:{(t_weight * tanh_loss).item():.3f}, LPR1:{loss_pr1.item():.3f}, LHO:{loss_ho.item():.3f}, decL:{dec_loss.item():.2f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac_aw:{acc_aw:.3f}, Ac_ag:{acc_ag:.3f}',refresh=False)            
                        self.logger.log_scalar(task= -1 if pretrain else t, iter=iter,name="LA",value=(align_pf_weight*a_loss_pf).item(),group='train')
                        self.logger.log_scalar(task= -1 if pretrain else t, iter=iter,name="LT",value=(t_weight * tanh_loss).item(),group='train')
                    self.logger.log_scalar(task=t, iter=iter,name="LC",value=(cl_weight*class_loss).item(),group='train')
                self.logger.log_scalar(task= -1 if pretrain else t, iter=iter, name="loss_pr1",value=loss_pr1.item(),group='pretrain' if pretrain else 'train')
                self.logger.log_scalar(task= -1 if pretrain else t, iter=iter, name="loss_ho",value=loss_ho.item(),group='pretrain' if pretrain else 'train')
                self.logger.log_scalar(task= -1 if pretrain else t, iter=iter, name="loss",value=loss.item(),group='pretrain' if pretrain else 'train')
                self.logger.log_scalar(task= -1 if pretrain else t, iter=iter, name="dec_loss",value=dec_loss.item(),group='pretrain' if pretrain else 'train')
        return loss, acc
    
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-8):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

    