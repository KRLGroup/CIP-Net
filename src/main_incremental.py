import argparse
import importlib
import os
import time
from functools import reduce

import numpy as np
import torch
import torch.nn as nn

import approach
import utils
from utils import create_plots, extract_log_from_raw
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from loggers.exp_logger import MultiLogger
from networks import allmodels
from networks.cipnet_utils.cipnet import CIPNet #Aggiunto
from networks.cipnet_utils.util.args import get_optimizer_nn, get_optimizer_newhead # aggiunto
from networks.cipnet_utils.util.data import get_dataloaders # aggiunto
from networks.cipnet_utils.util.func import init_weights_xavier
from wrapperDDP import wrapper
import torch.multiprocessing as mp   
from torch.distributed import init_process_group, destroy_process_group 

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args, extra_args):
    if rank is not None and world_size is not None:
        ddp_setup(rank, world_size)


    tstart = time.time()

    args.results_path = os.path.expanduser(args.results_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # random.seed(args.seed)
    np.random.seed(args.seed)

    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_net=args.lr_net, lr_block=args.lr_block, 
                        lr_min=args.lr_min, lr_factor=args.lr_factor, lr_patience=args.lr_patience,
                        clipgrad=args.clipping, momentum=args.momentum, wd=args.weight_decay, 
                        multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs, 
                        wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    print(torch.cuda.is_available())
    if torch.cuda.is_available() and args.gpu != "cpu":
        gpus = [int(g) for g in str(args.gpu).split(',')]
        # torch.cuda.set_device(gpus[0])
        if args.parallelization=="DDP":
            device = rank
        else:
            device = 'cuda:'+str(gpus[0])
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
        gpus = [device]
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Network
    if "cipnet" in args.network:
        args.network = 'convnext_tiny_26'
    print("USING:",args.network)
    print("Preparing CIPNet")
    cipnet_loss = None

    from networks.cipnet_utils.cipnet import construct_CIPNet
    classes_pertask = args.num_classes//args.num_tasks #args.num_classes is the total number of classes in the dataset
    init_model = construct_CIPNet(num_classes=classes_pertask,
                                    network=args.network,
                                    pretrained=args.pretrained,
                                    bias=args.bias,
                                )

    print("NUM PROTOTYPES", init_model._num_prototypes, flush=True)
    init_model = init_model.to(device)
    init_model = wrapper(init_model, devices = gpus, parallelization=args.parallelization, rank=rank)
    print(init_model.module)
    with torch.no_grad():
        if args.load_backbone != '':
            checkpoint = torch.load(args.load_backbone, map_location=device)
            init_model.module._net.load_state_dict(checkpoint['proto_pretrained_backbone_task_3'],strict=True) 
            print("backbone network loaded", flush=True)
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                print("Optimizer state dict not found in checkpoint")

        if args.load_classifiers != '':
            checkpoint = torch.load(args.load_classifiers, map_location=device)
            init_model.load_state_dict(checkpoint['model_state_dict'],strict=True)             
        else:
            init_model.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(init_model.module._classification[-1].weight, mean=1.0,std=0.1) 
            if args.bias:
                torch.nn.init.constant_(init_model.module._classification[-1].bias, val=0.)
            print("Classification layer initialized with mean", torch.mean(init_model.module._classification[-1].weight).item(), flush=True)

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr') #inizializza Inc_Learning_Appr
    assert issubclass(Appr, Inc_Learning_Appr)
    #ignore###
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)
    ##########

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_model)

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, _, taskcla, d_idxs = get_loaders(args.datasets, args.num_tasks,
                                                                          args.nc_first_task,
                                                                          args.batch_size, num_workers=args.num_workers,
                                                                          pin_memory=args.pin_memory,
                                                                          repeat_task_0=args.repeat_task_0,
                                                                          parallelization=args.parallelization)
    
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task
    
    #CIPNet L_C
    cipnet_loss = nn.NLLLoss(reduction='mean').to(device) #L_C

    # Network and Approach instances         
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
        
    utils.seed_everything(seed=args.seed)
    appr = Appr(model=init_model, device=device, network=args.network, cipnet_loss=cipnet_loss,
                use_hoyer=args.use_hoyer, use_proto_reg=args.use_proto_reg,
                proto_reg_CE=args.proto_reg_CE, **appr_kwargs)
    appr.lamb_proto = args.lamb_proto
    appr.lamb_hoyer = args.lamb_hoyer
    appr.results_path = args.results_path
    appr.exp_name = args.exp_name
    appr.num_tasks = args.num_tasks
    appr.classes_pertask = classes_pertask
    appr.proto_used = {k:torch.zeros(768) for k in range(args.num_tasks)}
    appr.datasetname = args.datasets[0]+"_"+str(args.approach)+"_"

    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    ppnet_losses = {}
    st = 0
    optimizer_classifiers = []
 
    with torch.no_grad():
        if args.load_model != '':
            checkpoint = torch.load(args.load_model, map_location=device)
            st = checkpoint['task']+1
            if checkpoint['task'] > 0:
                for j in range(checkpoint['task']):
                    appr.model.module.add_head(bias=args.bias)
            appr.model.module._net.load_state_dict(checkpoint['backbone'])
            appr.model.module._classification.load_state_dict(checkpoint['classifiers'])
            if 'tau' in checkpoint.keys():
                appr.model.module.tau = checkpoint['tau']
            else:
                appr.model.module._multipliers = [cls.normalization_multiplier for cls in appr.model.module._classification]

            appr.proto_used = checkpoint['proto_used']
            for k,v in appr.proto_used.items():
                appr.proto_used[k] = v.cpu()
            appr.proto_idxs = checkpoint['proto_idxs'].cpu()

            optimizer_classifiers = []
            scheduler_classifiers = []
            assert len(checkpoint['optimizer_classifier_state_dicts']) == checkpoint['task']+1
            for k, opt_state_dict in enumerate(checkpoint['optimizer_classifier_state_dicts']):
                classification_weight = []
                classification_bias = []
                for name, param in appr.model.module._classification[k].named_parameters():
                    if 'weight' in name:
                        classification_weight.append(param)
                    elif 'multiplier' in name:
                        param.requires_grad = False
            #         else:
            #             if args.bias:
            #                 classification_bias.append(param)
                        
                paramlist_classifier = [
                        {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
                        {"params": classification_bias, "lr": args.lr, "weight_decay_rate": args.weight_decay},
                ]
                    
                # if args.optimizer == 'Adam': # OPTIMIZER È ADAM
                optimizer_classifiers.append(torch.optim.AdamW(paramlist_classifier, lr=args.lr, weight_decay=args.weight_decay))
                optimizer_classifiers[k].load_state_dict(opt_state_dict)
                scheduler_classifiers.append(
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifiers[-1], T_0=5, eta_min=0.001, T_mult=1, verbose=False) if args.nepochs<=30 else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifiers[-1],T_0=10, eta_min=0.001, T_mult=1, verbose=False))

            appr.post_train_process(t=0)
            appr.model.to(device)
            appr.model_old.to(device)
            print("MODELLO CARICATO")


    for t in range(st,len(taskcla)):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # # Add head for current task
        init_model.to(device)
        if t>0:
            appr.model.module.add_head(bias=args.bias)

        #PRETRAINING
        if args.pretrain_epochs>0:
            print("#"*25)
            print("PRETRAINING")
            if args.pretrain_cipnet:
                print("PRETRAIN")
                appr.pretrain_cipnet(t, args.pretrain_epochs,
                                        args.freeze_epochs,
                                        trn_loader[t],
                                        device,
                                        args.save_pretraining,
                                        args.load_backbone,
                                        args.results_path, 
                                        args.exp_name)
            
                if args.save_pretraining:
                    appr.model.eval()
                    p = "./"+str(args.results_path)+"/cub_200_2011_cropped_"+str(args.exp_name)+"/models/"
                    if not os.path.exists(p):
                        p = "./"+str(args.results_path)+"/"+str(args.exp_name)+"/models/"
                    if t>0:
                        if hasattr(appr.model.module,'tau'):
                            torch.save({
                            'task':t,
                            'class_idxs':d_idxs,
                            'pretrained_for_nepochs': args.pretrain_epochs,
                            'proto_used': appr.proto_used,
                            'proto_idxs': appr.proto_idxs,
                            'tau':appr.model.module.tau,
                            'proto_pretrained_backbone': appr.model.module._net.state_dict(),
                            'optimizer_net_state_dict': optimizer_net.state_dict(),
                            'classifiers':appr.model.module._classification.state_dict(),
                            'optimizer_classifier_state_dicts': [optimizer_classifier.state_dict() for opt_cls in optimizer_classifiers],
                            },#optimizer backbone del training alla task t-1
                            os.path.join(p, 'backbone_pretrained_'+str(args.pretrain_epochs)+"ep_t"+str(t)))
                        else:
                            torch.save({
                                'task':t,
                                'class_idxs':d_idxs,
                                'pretrained_for_nepochs': args.pretrain_epochs,
                                'proto_used': appr.proto_used,
                                'proto_idxs': appr.proto_idxs,
                                'proto_pretrained_backbone': appr.model.module._net.state_dict(),
                                'optimizer_net_state_dict': optimizer_net.state_dict(),
                                'classifiers':appr.model.module._classification.state_dict(),
                                'optimizer_classifier_state_dicts': [optimizer_classifier.state_dict() for opt_cls in optimizer_classifiers],
                                },#optimizer backbone del training alla task t-1
                                os.path.join(p, 'backbone_pretrained_'+str(args.pretrain_epochs)+"ep_t"+str(t)))
                    else:
                        if hasattr(appr.model.module,'tau'):
                            torch.save({
                            'task':t,
                            'class_idxs':d_idxs,
                            'pretrained_for_nepochs': args.pretrain_epochs,
                            'proto_used': appr.proto_used,
                            'proto_idxs': appr.proto_idxs,
                            'tau':appr.model.module.tau,
                            'proto_pretrained_backbone': appr.model.module._net.state_dict(),
                            'classifiers':appr.model.module._classification.state_dict()},
                            #optimize backbone del training non salvato perche in questo caso è stato fatto solo il pretraining della task 0
                            os.path.join(p, 'backbone_pretrained_'+str(args.pretrain_epochs)+"ep_t"+str(t)))
                        else:
                            torch.save({
                                'task':t,
                                'class_idxs':d_idxs,
                                'pretrained_for_nepochs': args.pretrain_epochs,
                                'proto_used': appr.proto_used,
                                'proto_idxs': appr.proto_idxs,
                                'proto_pretrained_backbone': appr.model.module._net.state_dict(),
                                'classifiers':appr.model.module._classification.state_dict()},
                                #optimize backbone del training non salvato perche in questo caso è stato fatto solo il pretraining della task 0
                                os.path.join(p, 'backbone_pretrained_'+str(args.pretrain_epochs)+"ep_t"+str(t)))

                    appr.model.train()
       
                print("FINE PRETRAINING")
        # SECOND TRAINING PHASE
        # re-initialize optimizers and schedulers for second training phase [in this case for each task]
        if t == 0:
            optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = appr.get_optimizer_nn() 
            scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net,T_max=len(trn_loader[t])*args.nepochs,eta_min=args.lr_net/100.)
            # scheduler for the classification layer is with restarts, such that the model
            # can re-active zeroed-out prototypes. Hence an intuitive choice. 
            optimizer_classifiers = [optimizer_classifier]
            
            if args.nepochs<=30:
                scheduler_classifiers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifiers[-1], 
                                                                                            T_0=5, eta_min=0.001, T_mult=1, verbose=False)]
            else:
                scheduler_classifiers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifiers[-1],
                                                                                        T_0=10, eta_min=0.001, T_mult=1, verbose=False)]

        elif t>0:
            # del optimizer_net
            # del scheduler_net
            optimizer_net, _, params_to_freeze, params_to_train, params_backbone = appr.get_optimizer_nn()
            scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net,T_max=len(trn_loader[t])*args.nepochs,eta_min=args.lr_net/100.) 
            optimizer_classifier_newhead = appr.get_optimizer_newhead() #check this
            optimizer_classifiers.append(optimizer_classifier_newhead)
            print("LEN OPTIMIZER CLS:",len(optimizer_classifiers))
            if args.nepochs<=30:
                scheduler_classifiers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                                    optimizer_classifiers[-1],
                                                                    T_0=5,
                                                                    eta_min=0.001,
                                                                    T_mult=1,
                                                                    verbose=False))
            else:
                scheduler_classifiers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                                    optimizer_classifiers[-1],
                                                                    T_0=10,
                                                                    eta_min=0.001,
                                                                    T_mult=1,
                                                                    verbose=False))
        

        for param in appr.model.module.parameters():
            param.requires_grad = False
        l_cls = len(appr.model.module._classification)
        for i in range(l_cls):
            for param in appr.model.module._classification[i].parameters():
                if i!=l_cls-1:
                    param.requires_grad = False
                elif i==l_cls-1:
                    param.requires_grad = True
                    
        val_acc = appr.train(t, trn_loader[t], val_loader[t], None, args.pretrain_epochs, args.freeze_epochs, params_to_freeze,
                    params_to_train, params_backbone, optimizer_net, optimizer_classifiers, scheduler_net,
                    scheduler_classifiers, device)
        
        print(val_acc)
        print('-' * 108)
        print('#' * 108)        
        print("TEST")

        for u in range(t+1):
            print("TEST TASK",u, flush=True)
            tst_info = appr.eval_test(t, u, tst_loader[u], device, args.results_path, args.exp_name, appr.classes_pertask)
        print("TEST FINISHED")
        print('#' * 108)      
        print('-' * 108)
     
        with torch.no_grad():
            appr.model.eval()
            # if self.model.rank == 0 or self.model.rank == -1:
            if args.save_model:
                if hasattr(appr.model.module, 'tau'):
                    torch.save({
                        'task':t,
                        'class_idxs':d_idxs,
                        'proto_used': appr.proto_used,
                        'proto_idxs': appr.proto_idxs,
                        'tau':appr.model.module.tau, # tau is shared parameter in this case and saved with the _net state dict 
                        'backbone': appr.model.module._net.state_dict(),
                        'classifiers': appr.model.module._classification.state_dict(),
                        'optimizer_net_state_dict': optimizer_net.state_dict(),
                        'optimizer_classifier_state_dicts': [opt_cls.state_dict() for opt_cls in optimizer_classifiers],
                    }, os.path.join("./"+str(args.results_path)+"/"+str(appr.datasetname)+str(args.exp_name)+"/models/", "model_task"+str(t)+".pt"))
                elif hasattr(appr.model.module._classification[-1], 'tau'):
                    torch.save({
                        'task':t,
                        'class_idxs':d_idxs,
                        'proto_used': appr.proto_used,
                        'proto_idxs': appr.proto_idxs, 
                        'backbone': appr.model.module._net.state_dict(),
                        'classifiers': appr.model.module._classification.state_dict(), # each had has a learnable tau, it is saved here
                        'optimizer_net_state_dict': optimizer_net.state_dict(),
                        'optimizer_classifier_state_dicts': [opt_cls.state_dict() for opt_cls in optimizer_classifiers],
                    }, os.path.join("./"+str(args.results_path)+"/"+str(appr.datasetname)+str(args.exp_name)+"/models/", "model_task"+str(t)+".pt"))
                else:
                    torch.save({
                        'task':t,
                        'class_idxs':d_idxs,
                        'proto_used': appr.proto_used,
                        'proto_idxs': appr.proto_idxs,
                        # 'tau':appr.model.module.tau, # tau is shared parameter in this case and saved with the _net state dict 
                        'normalizers': [m for m in appr.model.module._multiplier],
                        'backbone': appr.model.module._net.state_dict(),
                        'classifiers': appr.model.module._classification.state_dict(),
                        'optimizer_net_state_dict': optimizer_net.state_dict(),
                        'optimizer_classifier_state_dicts': [opt_cls.state_dict() for opt_cls in optimizer_classifiers],
                    }, os.path.join("./"+str(args.results_path)+"/"+str(appr.datasetname)+str(args.exp_name)+"/models/", "model_task"+str(t)+".pt"))
    
    
    # Create graphs
    d_log = extract_log_from_raw(args.results_path, args.exp_name, args.num_tasks, appr.datasetname)
    create_plots(d_log, appr.proto_used , args.results_path, args.exp_name, appr.datasetname)
    print("PLOTS CREATED")
    print('Done!')
    if rank is not None and world_size is not None:
        destroy_process_group()
    exit()

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
#     ####################################################################################################################

import trace
if __name__ == '__main__':
    # tracer = trace.Trace(count=False, trace=True)
    # tracer.run('main()')
    import sys

    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=str, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk', 'tensorboard'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    # parser.add_argument('--save-models', action='store_true',
    #                     help='Save trained models (default=%(default)s)')
    # parser.add_argument('--last-layer-analysis', action='store_true',
    #                     help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num_tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='convnext_tiny_26', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    # parser.add_argument('--keep-existing-head', action='store_true',
    #                     help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-6, type=float, required=False,
                        help='ICICLE, Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='ICICLE, Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='ICICLE, Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='ICICLE, Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='ICICLE, Momentum factor (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='ICICLE, Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='ICICLE, Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='ICICLE, Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='ICICLE, Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')

    parser.add_argument('--num_classes', type=int, help='Number of classes in the whole dataset')
    parser.add_argument('--repeat_task_0', action='store_true', help='Repeat task 0')
    parser.add_argument('--bias', action='store_true', help='bias')

    # ADDED FOR CIPNET
    parser.add_argument('--cipnetdataset', default='CUB-200-2011', help='dataset for pretraining')
    parser.add_argument('--pretrain_epochs', type=int, default=0, help='epochs to pretrain prototypes')
    parser.add_argument('--freeze_epochs', type=int, default=0, help='number of epochs to keep the backbone frozen at the start of the training phase')
    parser.add_argument('--save_model', action='store_true', help='save complete model')
    parser.add_argument('--save_pretraining', action='store_true', help='save model just after pretraining')
    parser.add_argument('--load_backbone', type=str, default='', help='path to checkpoint for pipnet')
    parser.add_argument('--load_classifiers', type=str, default='', help='path to checkpoint for pipnet')
    parser.add_argument('--load_model', type=str, default='', help='path to checkpoint for pipnet')
    parser.add_argument('--use_hoyer', action='store_true', help='use hoyer loss')
    parser.add_argument('--lamb_hoyer', default=10.0, type=float, required=False, help='The optimizer learning rate for training the weights from prototypes to classes (default=%(default)s)')
    parser.add_argument('--use_proto_reg', action='store_true', help='use prototype regularization')
    parser.add_argument('--lamb_proto', default=0.005, type=float, required=False, help='The optimizer learning rate for training the weights from prototypes to classes (default=%(default)s)')
    parser.add_argument('--proto_reg_CE', action='store_true', help='use prototype regularization with CROSS ENTROPY')
    parser.add_argument('--pretrain_cipnet', action='store_true', help='use pretrain cipnet')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='The optimizer learning rate for training the weights from prototypes to classes (default=%(default)s)')
    parser.add_argument('--lr_net', default=0.0005, type=float, required=False, help='The optimizer learning rate for the backbone. Usually similar as lr_block. (default=%(default)s)')
    parser.add_argument('--lr_block', default=0.0005, type=float, required=False, help='The optimizer learning rate for training the last conv layers of the backbone (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False, help='Weight decay used in the optimizer (default=%(default)s)')
    # parser.add_argument('--gridsearch-njobs', type=int, default=1, help='number of jobs for gridsearch')
    parser.add_argument('--parallelization', type=str, default='DP', help='type of parallelization, for no parallelization set "NO"')


    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(sys.argv)
    extra_args = extra_args[1:]
    
    if args.parallelization=="DDP":
        gpus = [int(g) for g in str(args.gpu).split(',')]
        world_size = len(gpus)
        mp.spawn(main, args=(world_size, args, extra_args), nprocs=world_size)
    else:
        main(rank=None, world_size=None, args=args, extra_args=extra_args)
