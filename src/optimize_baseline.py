import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# mp.set_start_method('fork', force=True)
# import multiprocessing.process as _mp_proc
# _mp_proc.BaseProcess.daemon = property(lambda self: False, lambda self, val: None)
import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real, Integer, Categorical
# from skopt.utils import use_named_args
import argparse
from networks.pipnet_utils.pipnet import construct_PIPNet
import importlib
import torch
import torch.nn as nn
import numpy as np
from datasets.data_loader import get_loaders
from networks.pipnet_utils.util.func import init_weights_xavier
from datasets.dataset_config import dataset_config
from networks import allmodels
from wrapperDDP import wrapper
from functools import reduce
from loggers.exp_logger import MultiLogger
from utils import create_plots, extract_log_from_raw

import itertools
from joblib import Parallel, delayed
import random
import gc
import pickle
import os

class CustomGridSearch:
    def __init__(self, fixed_params, param_grid, n_jobs=1, device='cpu'):
        # self.model = model
        self.fixed_params = fixed_params
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []
        self.device = device
        

    def fit(self):    
        # Generate all possible combinations of parameters
        param_combinations = list(itertools.product(*(self.param_grid[param] for param in self.param_grid)))
        random.shuffle(param_combinations)
        
        # for c in param_combinations:
        #     if c[0] <= c[6]: #rimuove combinazioni in cui freeze epochs > nepochs
        #         param_combinations.remove(c)

        print('Total combinations:', len(param_combinations))

        # Parallel processing of parameter combinations
        results = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(self.evaluate_params)(params) for params in param_combinations)
        
        # Process results to find the best parameters and score
        for param_dict, score in results:
            print(param_dict, score)
            avg_score = torch.mean(torch.tensor(score))
            self.results_.append({'params': param_dict, 'score': score, "avg":avg_score})
            if self.best_score_ is None or avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = param_dict
                print("CURRENT BEST PARAM:")
                print(self.best_params_)
        
        return self
    
    def evaluate_params(self, params):
        Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning_pipnet'), 'FT')
        
        param_dict = {param: params[i] for i, param in enumerate(self.param_grid)}
        full_exp_name = self.fixed_params['exp_name']+"/" #dovrebbe essere la data
        for k,v in param_dict.items():
            s = str(k)
            sv = str(v)
            s = k.split('_')
            if len(s)>1:
                s = "".join([h[0] for h in s])
            else:
                s = s[0] 
            sv.replace(".","")

            full_exp_name += str(s)+'-'+str(sv)+'_'
        print(param_dict)
        param_dict.update(self.fixed_params)

        torch.manual_seed(param_dict['seed'])
        # torch.cuda.manual_seed_all(param_dict['seed'])
        random.seed(param_dict['seed'])
        np.random.seed(param_dict['seed'])

        logger = MultiLogger(param_dict['results_path'], full_exp_name, loggers=param_dict['log'], save_models=param_dict['save_model'])
        # param_dict['logger'] = logger
        # logger = None

        model = construct_PIPNet(num_classes=param_dict['num_classes'], network=param_dict['network'], pretrained=param_dict['pretrained'], bias=param_dict['bias'], use_heads=True)

        model = wrapper(model, devices = param_dict['gpus'], parallelization=param_dict['parallelization'])

        appr_ft = Appr_finetuning(model, self.device)
        for k, v in param_dict.items():
            setattr(appr_ft, k, v)
            # print("attr:", k, getattr(appr_ft,k))
        setattr(appr_ft, 'exp_name', full_exp_name)
        print("appr device:", appr_ft.device)
        appr_ft.logger = logger

        def getdataloaders(params):
            return  get_loaders(params['dataset'],
                                4,
                                None,
                                params['batch_size'],
                                num_workers=params['num_workers'],
                                pin_memory=True,
                                repeat_task_0=False,
                                use_pipnet=True,######## aggiunto
                                parallelization=params['parallelization']) ######## aggiunto

        trn_loader, val_loader, tst_loader, _, _ = getdataloaders(param_dict)
        appr_ft.model.to(self.device)
        if param_dict['startfrom2nd'] and param_dict['path_to_model']!='':
            ts = 1
            checkpoint = torch.load(param_dict['path_to_model'], map_location=self.device)
            appr_ft.model.module._net.load_state_dict(checkpoint['backbone'])
            appr_ft.model.module._classification.load_state_dict(checkpoint['classifiers'])
            if 'tau' in checkpoint.keys():
                appr_ft.model.module.tau = checkpoint['tau']
            else:
                appr_ft.model.module._multipliers = [cls.normalization_multiplier for cls in appr_ft.model.module._classification]

            appr_ft.proto_used = checkpoint['proto_used']
            for k,v in appr_ft.proto_used.items():
                appr_ft.proto_used[k] = v.cpu()
            appr_ft.proto_idxs = checkpoint['proto_idxs'].cpu()
            
            optimizer_classifiers = [appr_ft.get_optimizer_newhead()]
            scheduler_classifiers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifiers[-1], 
                                                                                            T_0=5, eta_min=0.001, T_mult=1, verbose=False)]
            print("MODELLO CARICATO")
            appr_ft.post_train_process(t=0)
            appr_ft.model.to(self.device)
            appr_ft.model_old.to(self.device)
        else:
            ts = 0
            appr_ft.model.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(appr_ft.model.module._classification[-1].weight, mean=1.0,std=0.1) 
            if param_dict['bias']:
                torch.nn.init.constant_(appr_ft.model.module._classification[-1].bias, val=0.)
            #uncomment if cosinelinear head not used
            # torch.nn.init.constant_(model.module._multiplier[-1], val=2.)
            # model.module._multiplier[-1].requires_grad = False
            print("Classification layer initialized with mean", torch.mean(appr_ft.model.module._classification[-1].weight).item(), flush=True)
        
        for t in range(ts,2):#Per velocizzare traino solo fino alla seconda task
            appr_ft.model.train()
            if t>0:
                appr_ft.model.module.add_head(bias=False)
                appr_ft.model.to(self.device)

            
            #Pretrain
            if param_dict['pretrain_epochs']>0:
                print("pretrain started for task:", t)
                appr_ft.pretrain_icicle(t, param_dict['pretrain_epochs'], param_dict['freeze_epochs'], trn_loader[t], self.device, 
                                    save_model=True, load_backbone='', results_path=param_dict['results_path'], exp_name=full_exp_name)
            
            if t == 0:
                optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = appr_ft.get_optimizer_nn()       
                scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net,T_max=len(trn_loader[t])*param_dict['nepochs'],eta_min=param_dict['lr_net']/100.)
                # scheduler for the classification layer is with restarts, such that the model
                # can re-active zeroed-out prototypes. Hence an intuitive choice. 
                optimizer_classifiers = [optimizer_classifier]
                if param_dict['nepochs']<=30:
                    scheduler_classifiers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifiers[-1], 
                                                                                                T_0=5, eta_min=0.001, T_mult=1, verbose=False)]
                else:
                    scheduler_classifiers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifiers[-1],
                                                                                            T_0=10, eta_min=0.001, T_mult=1, verbose=False)]

            elif t>0:
                # del optimizer_net
                # del scheduler_net
                optimizer_net, _, params_to_freeze, params_to_train, params_backbone = appr_ft.get_optimizer_nn()
                scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net,T_max=len(trn_loader[t])*param_dict['nepochs'],eta_min=param_dict['lr_net']/100.) 
                optimizer_classifier_newhead = appr_ft.get_optimizer_newhead() #check this
                optimizer_classifiers.append(optimizer_classifier_newhead)
                if param_dict['nepochs']<=30:
                    scheduler_classifiers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                                        optimizer_classifiers[-1],
                                                                        T_0=5,
                                                                        eta_min=0.001,
                                                                        T_mult=1,
                                                                        verbose=False))
                else:
                    scheduler_classifiers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                                        optimizer_classifiers[1],
                                                                        T_0=10,
                                                                        eta_min=0.001,
                                                                        T_mult=1,
                                                                        verbose=False))
    
            for param in appr_ft.model.module.parameters():
                param.requires_grad = False
                
            l_cls = len(appr_ft.model.module._classification)
            for i in range(l_cls):
                for param in appr_ft.model.module._classification[i].parameters():
                    if i!=l_cls-1:
                        param.requires_grad = False
                    elif i==l_cls-1:
                        param.requires_grad = True

            val_acc = appr_ft.train(t, trn_loader[t], val_loader[t], None, param_dict['pretrain_epochs'], param_dict['freeze_epochs'], params_to_freeze,
                    params_to_train, params_backbone, optimizer_net, optimizer_classifiers, scheduler_net,
                    scheduler_classifiers, self.device)

            if len(val_acc)>0:
                print("val_acc")
                print(val_acc)
            score = []
            with torch.no_grad():
                appr_ft.model.eval()
                if hasattr(appr_ft.model.module, 'tau'):
                    torch.save({
                        'task':t,
                        'tau':appr_ft.model.module.tau,
                        'proto_used': appr_ft.proto_used,# non c'erano prima
                        'proto_idxs': appr_ft.proto_idxs,# non c'erano prima
                        'backbone': appr_ft.model.module._net.state_dict(),
                        'classifiers': appr_ft.model.module._classification.state_dict(),
                        'optimizer_net_state_dict': optimizer_net.state_dict(),
                        'optimizer_classifier_state_dicts': [opt_cls.state_dict() for opt_cls in optimizer_classifiers],
                    }, os.path.join("./"+str(param_dict['results_path'])+"/"+full_exp_name+"/models/", "model_at_task"+str(t)+".pt"))
                else:
                    torch.save({
                        'task':t,
                        'normalizers': [m for m in appr_ft.model.module._multiplier],
                        'proto_used': appr_ft.proto_used,# non c'erano prima
                        'proto_idxs': appr_ft.proto_idxs,# non c'erano prima
                        'backbone': appr_ft.model.module._net.state_dict(),
                        'classifiers': appr_ft.model.module._classification.state_dict(),
                        'optimizer_net_state_dict': optimizer_net.state_dict(),
                        'optimizer_classifier_state_dicts': [opt_cls.state_dict() for opt_cls in optimizer_classifiers],
                    }, os.path.join("./"+str(param_dict['results_path'])+"/"+full_exp_name+"/models/", "model_at_task"+str(t)+".pt"))


        with torch.no_grad():
            for u in range(2):
                print("TEST TASK",u, flush=True)
                tst_info = appr_ft.eval_test(1, u, tst_loader[u], self.device, param_dict['results_path'], full_exp_name, use_heads=True)
                score.append(tst_info['test_accuracy_task'])
        
        # with torch.no_grad():
        #     appr_ft.model.eval()

        #     torch.save({
        #         'backbone': appr_ft.model.module._net.state_dict(),
        #         'classifiers': appr_ft.model.module._classification.state_dict(),
        #         'optimizer_net_state_dict': optimizer_net.state_dict(),
        #         'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
        #     }, os.path.join("./"+str(param_dict['results_path'])+"/"+full_exp_name+"/models/", "complete_model.pt"))
            
        #MODIFICARE PER IL CASO IN CUI from2ndtask è true
        pp = param_dict['path_to_model'].split('/')[:-2]
        d_log = extract_log_from_raw("./"+pp[1], '/'.join(pp[2:]))
        d_log1 = extract_log_from_raw(param_dict['results_path'], full_exp_name)
        def merge_logs(d1, d2):
            for group_key, group_val in d2.items(): # 'train', {0:{},...}
                if isinstance(group_val, dict):
                    for sub_key, sub_val in group_val.items():# 0, {'LA':[]}
                        if isinstance(sub_val, dict):  # Nested task structure (e.g., train/val)
                            for name, values in sub_val.items():# 'LA', []
                                if name in d1[group_key][sub_key]:
                                    if sub_key == 1:
                                        d1[group_key][sub_key][name] = []
                                    d1[group_key][sub_key][name].extend(values)
                                else:
                                    print("IMPOSSIBLE")
                                    # d1[group_key][sub_key][name] = values.copy()
                        else:  # Flat metric structure (e.g., test/pretrain)
                            if sub_key in d1[group_key]:
                                d1[group_key][sub_key].extend(sub_val)
                            else:
                                d1[group_key][sub_key] = sub_val.copy()
                # else:
                #     # In case there's a flat list or other structure (not expected in current design)
                #     d1[group_key] = group_val
            return d1
        d_log = merge_logs(d_log,d_log1)
        

        create_plots(d_log, appr_ft.proto_used , param_dict['results_path'], full_exp_name)
        print("PLOTS CREATED")
        del trn_loader, val_loader, tst_loader
        gc.collect()
        torch.cuda.empty_cache()
        return param_dict, score


    def get_results(self):
        return self.results_





# Define the search space for hyperparameters
# params  = {
#     'nogumbel': [False, True],
#     'epochs': [3000],
#     'lr': [0.001, 0.01, 0.1],
#     'l2': [0, 0.001, 0.01, 0.1],
#     'dropout': [0, 0.15, 0.3],
#     'num_layers': [3],
#     'hidden_dim': [16,32,64,128],
#     'batch_size': [16,32,64],
# }
# params = {
#             'nepochs': [1],
#             'lr': [0.05],
#             'lr_net':[0.0005],
#             'lr_block':[0.0005, 0.01],
#             'batch_size': [64],
#             'pretrain_epochs':[0],
#             'freeze_epochs':[1],#check come gestire freeze epochs se epochs sono meno
#             'proto_reg_CE':[True,False],
#             'seed':[1]

#         }

# params = {
#             'nepochs': [2, 5, 10, 20],
#             'lr': [0.01,0.001,0.0001],
#             'lr_net':[0.0001, 0.01, 0.001],
#             'lr_block':[0.0001, 0.01, 0.001],
#             'batch_size': [64],
#             'pretrain_epochs':[5, 10],
#             'freeze_epochs':[1, 3, 5],
#             'lamb_proto':[0.01, 0.001, 0.0001],
#             'lamb_hoyer':[10.0, 100.0],
#             'proto_reg_CE':[False],
#             'wd':[0.0],
#             'seed':[1]

#         }


params = {
            'nepochs': [20],
            'lr': [0.001],
            'lr_net':[0.0001],
            'lr_block':[0.0001],
            'batch_size': [64],
            'pretrain_epochs':[10],
            'freeze_epochs':[1],
            'lamb_proto':[1.3, 1.5, 1.7, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0],
            'lamb_hoyer':[10.0],
            'proto_reg_CE':[False],
            'wd':[0.0],
            'seed':[1]

        }




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize_baseline.py')
    parser.add_argument('--cipnetdataset', default='CUB-200-2011', help='FOR PIPNET: dataset for pretraining')
    parser.add_argument('--use_heads', action='store_true', help='Use heads with pipnet')
    parser.add_argument('--pretrain_epochs', type=int, default=0, help='epochs to pretrain prototypes')
    parser.add_argument('--freeze_epochs', type=int, default=0, help='number of epochs to keep the backbone frozen at the start of the training phase')
    parser.add_argument('--save_model', action='store_true', help='path to checkpoint for pipnet')
    parser.add_argument('--load_backbone', type=str, default='', help='path to checkpoint for pipnet')
    parser.add_argument('--load_classifiers', type=str, default='', help='path to checkpoint for pipnet')
    parser.add_argument('--load_model', type=str, default='', help='path to checkpoint for pipnet')
    parser.add_argument('--use_hoyer', action='store_true', help='use hoyer loss')
    parser.add_argument('--lamb_hoyer', default=10.0, type=float, required=False, help='The optimizer learning rate for training the weights from prototypes to classes (default=%(default)s)')
    parser.add_argument('--use_proto_reg', action='store_true', help='use prototype regularization')
    # parser.add_argument('--lamb_proto', default=0.005, type=float, required=False, help='The optimizer learning rate for training the weights from prototypes to classes (default=%(default)s)')
    parser.add_argument('--proto_reg_CE', action='store_true', help='use prototype regularization with CROSS ENTROPY')
    parser.add_argument('--pretrain_icicle', action='store_true', help='use pretrain icicle (dataset della task specifica)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='The optimizer learning rate for training the weights from prototypes to classes (default=%(default)s)')
    parser.add_argument('--lr_net', default=0.0005, type=float, required=False, help='The optimizer learning rate for the backbone. Usually similar as lr_block. (default=%(default)s)')
    parser.add_argument('--lr_block', default=0.0005, type=float, required=False, help='The optimizer learning rate for training the last conv layers of the backbone (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False, help='Weight decay used in the optimizer (default=%(default)s)')
    parser.add_argument('--parallelization', type=str, default='DP', help='type of parallelization, for no parallelization set "NO"')

    parser.add_argument('--startfrom2nd', action='store_true', help='load model trained on first task and start directly with second task for all combinations')
    parser.add_argument('--results-path', type=str, default='../results', help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str, help='Experiment name (default=%(default)s)')
    parser.add_argument('--log', default=['disk', 'tensorboard'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--dataset',  default=['cub_200_2011_cropped'],type=str, choices=list(dataset_config.keys()),
                            help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--n_jobs',  default=10, type=int, help='Number of jobs')
    parser.add_argument('--network', default='convnext_tiny_26', type=str, help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--num_classes', type=int, help='Number of classes in the whole dataset')
    parser.add_argument('--bias', action='store_true', help='bias')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--gpu', type=str, default=0, help='GPU (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone (default=%(default)s)')
    args = parser.parse_args()
    dataset_name = args.dataset
    n_jobs = args.n_jobs
    if args.parallelization=="DP":
        gpus = [int(g) for g in str(args.gpu).split(',')]
        device = 'cuda:'+str(gpus[0])
    

    fixed_params = dict(results_path=args.results_path,
                        exp_name=args.exp_name,
                        log=args.log,
                        network=args.network,
                        pretrained=True, #pretrained backbone on imagenet
                        dataset=args.dataset,
                        num_workers=args.num_workers,
                        use_pipnet=True,
                        pipnet_loss=nn.NLLLoss(reduction='mean'),
                        use_heads=True,
                        use_hoyer=True,
                        use_proto_reg=True,
                        save_model=args.save_model,
                        parallelization=args.parallelization,
                        num_classes=args.num_classes,
                        bias=args.bias,
                        gpus=gpus,
                        startfrom2nd = args.startfrom2nd,
                        path_to_model=args.load_model
                        )
    
    grid_search = CustomGridSearch(fixed_params, params, n_jobs=n_jobs, device=device)
    grid_search.fit()
    
    # Retrieve results
    results = grid_search.get_results()
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    # print("All Results:", results)

    