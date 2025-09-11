from tqdm import tqdm
import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from networks.cipnet_utils.util.log import Log
from networks.cipnet_utils.util.func import topk_accuracy
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score

@torch.no_grad()
def eval_cipnet(net, 
        task,
        u_task,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,  
        progress_prefix: str = 'Eval Epoch',
        classes_per_task = 50,
        ) -> dict:
    # print(type(net))
    net = net.to(device)
    # print(type(net))
    # Make sure the model is in evaluation mode
    net.eval()
    # print(type(net))
    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    num_tasks = 200//classes_per_task
    cm_total = np.zeros((200, 200), dtype=int)
    cm_task = np.zeros((classes_per_task, classes_per_task), dtype=int)
    cm_task_ag = np.zeros((net.module._num_classes*num_tasks, net.module._num_classes*num_tasks), dtype=int)

    global_top1acc = 0.
    global_top5acc = 0.
    global_sim_anz = 0.
    global_anz = 0.
    local_size_total = 0.
    y_trues = []
    y_preds = []
    y_preds_classes = []
    abstained = 0
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        mininterval=5.,
                        ncols=0)
    (xs, ys) = next(iter(test_loader))
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        # print("xs shape:", xs.shape)
        # print("EVAL X:",xs)
        # if i == 2:
        #     break
        xs, ys = xs.to(device), ys.to(device)
        with torch.no_grad():
            # net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) 
            # Use the model to classify this batch of input data
            # _, pooled, out = net(xs, inference=True, task = None)
            _, pooled, out = net(xs, True, None) # (xs, inference, task)

            start = u_task * classes_per_task
            end = (u_task + 1) * classes_per_task
            max_out_score, ys_pred = torch.max(out[:,start:end], dim=1)
            _, ys_pred_ = torch.max(out, dim=1)

        
            # print("out",out.shape)
            out = out[:,start:end]
            ys_pred_scores = torch.amax(F.softmax(out,dim=1),dim=1)
            repeated_weight = net.module._classification[u_task].weight.unsqueeze(1).repeat(1,pooled.shape[0],1)
            
            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))
            sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(pooled*repeated_weight), 1e-3).float(),dim=2).float()
            local_size = torch.count_nonzero(torch.gt(torch.relu((pooled*repeated_weight)-1e-3).sum(dim=1), 0.).float(),dim=1).float()
            local_size_total += local_size.sum().item()

            correct_class_sim_scores_anz = torch.diagonal(torch.index_select(sim_scores_anz, dim=0, index=ys_pred),0)
            global_sim_anz += correct_class_sim_scores_anz.sum().item()
            
            almost_nz = torch.count_nonzero(torch.gt(torch.abs(pooled), 1e-3).float(),dim=1).float()
            global_anz += almost_nz.sum().item()
            
            # Update the confusion matrix
            cm_batch = np.zeros((classes_per_task, classes_per_task), dtype=int)
            for y_pred, y_pred_, y_true in zip(ys_pred, ys_pred_, ys):
                cm_task[y_true%classes_per_task][y_pred] += 1
                # TASK AGNOSTIC
                cm_task_ag[y_true][y_pred_] += 1
                cm_batch[y_true%classes_per_task][y_pred] += 1
            acc = acc_from_cm(cm_batch)
            test_iter.set_postfix_str(
                f'SimANZCC: {correct_class_sim_scores_anz.mean().item():.3f}, ANZ: {almost_nz.mean().item():.2f}, LocS: {local_size.mean().item():.2f}, Acc: {acc:.4f}', refresh=False
            )    
           
            (top1accs, top5accs) = topk_accuracy(out, ys, topk=[1,5])
            
            global_top1acc+=torch.sum(top1accs).item()
            global_top5acc+=torch.sum(top5accs).item()
            y_preds += ys_pred_scores.detach().tolist()
            y_trues += ys.detach().tolist()
            y_preds_classes += ys_pred.detach().tolist()
        
        del out
        del pooled
        del ys_pred
        
    print("CIP-Net abstained from a decision for", abstained.item(), "images", flush=True)      
    info['num non-zero prototypes'] = torch.gt(net.module._classification[u_task].weight,1e-3).any(dim=0).sum().item()
    info['confusion_matrix_task'] = cm_task
    info['confusion_matrix_task_ag'] = cm_task_ag
    info['confusion_matrix_total'] = None
    info['test_accuracy_task'] = acc_from_cm(cm_task)
    info['test_accuracy_task_ag'] = acc_from_cm(cm_task_ag)
    info['test_accuracy_total'] = None
    info['top1_accuracy'] = global_top1acc/len(test_loader.dataset)
    info['almost_sim_nonzeros'] = global_sim_anz/len(test_loader.dataset)
    info['local_size_all_classes'] = local_size_total / len(test_loader.dataset)
    info['almost_nonzeros'] = global_anz/len(test_loader.dataset)
    info['top5_accuracy'] = global_top5acc/len(test_loader.dataset) 

    return info

def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total


@torch.no_grad()
# Calculates class-specific threshold for the FPR@X metric. Also calculates threshold for images with correct prediction (currently not used, but can be insightful)
def get_thresholds(net,
        test_loader: DataLoader,
        epoch,
        device,
        percentile:float = 95.,
        log: Log = None,  
        log_prefix: str = 'log_eval_epochs', 
        progress_prefix: str = 'Get Thresholds Epoch',
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
    
    outputs_per_class = dict()
    outputs_per_correct_class = dict()
    for c in range(net.module._num_classes*4):
        outputs_per_class[c] = []
        outputs_per_correct_class[c] = []
    # Show progress on progress bar
    for t in range(4):
        test_iter = tqdm(enumerate(test_loader[t]),
                            total=len(test_loader[t]),
                            desc=progress_prefix+' %s Perc %s'%(epoch,percentile),
                            mininterval=5.,
                            ncols=0)
        (xs, ys) = next(iter(test_loader[t]))
        # Iterate through the test set
        for i, (xs, ys) in test_iter:
            xs, ys = xs.to(device), ys.to(device)
            
            with torch.no_grad():
                # Use the model to classify this batch of input data
                _, pooled, out = net(xs, inference=True)
                
                ys_pred = torch.argmax(out, dim=1)
                print(out.shape)
                for pred in range(len(ys_pred)):
                    outputs_per_class[ys_pred[pred].item()].append(out[pred,:].max().item())
                    if ys_pred[pred].item()==ys[pred].item():
                        outputs_per_correct_class[ys_pred[pred].item()].append(out[pred,:].max().item())
            
            del out
            del pooled
            del ys_pred

    class_thresholds = dict()
    correct_class_thresholds = dict()
    all_outputs = []
    all_correct_outputs = []
    for c in range(net.module._num_classes*4):
        if len(outputs_per_class[c])>0:
            outputs_c = outputs_per_class[c]
            all_outputs += outputs_c
            class_thresholds[c] = np.percentile(outputs_c,100-percentile) 
            
        if len(outputs_per_correct_class[c])>0:
            correct_outputs_c = outputs_per_correct_class[c]
            all_correct_outputs += correct_outputs_c
            correct_class_thresholds[c] = np.percentile(correct_outputs_c,100-percentile)
    
    overall_threshold = np.percentile(all_outputs,100-percentile)
    overall_correct_threshold = np.percentile(all_correct_outputs,100-percentile)
    # if class is not predicted there is no threshold. we set it as the minimum value for any other class 
    mean_ct = np.mean(list(class_thresholds.values()))
    mean_cct = np.mean(list(correct_class_thresholds.values()))
    for c in range(net.module._num_classes*4):
        if c not in class_thresholds.keys():
            print(c,"not in class thresholds. Setting to mean threshold", flush=True)
            class_thresholds[c] = mean_ct
        if c not in correct_class_thresholds.keys():
            correct_class_thresholds[c] = mean_cct

    calculated_percentile = 0
    correctly_classified = 0
    total = 0
    for c in range(net.module._num_classes*4):
        correctly_classified+=sum(i>class_thresholds[c] for i in outputs_per_class[c])
        total += len(outputs_per_class[c])
    calculated_percentile = correctly_classified/total

    if percentile<100:
        while calculated_percentile < (percentile/100.):
            class_thresholds.update((x, y*0.999) for x, y in class_thresholds.items())
            correctly_classified = 0
            for c in range(net.module._num_classes*4):
                correctly_classified+=sum(i>=class_thresholds[c] for i in outputs_per_class[c])
            calculated_percentile = correctly_classified/total

    return overall_correct_threshold, overall_threshold, correct_class_thresholds, class_thresholds

@torch.no_grad()
def eval_ood(net,
        test_loader: DataLoader,
        epoch,
        device,
        threshold, #class specific threshold or overall threshold. single float is overall, list or dict is class specific 
        progress_prefix: str = 'Get Thresholds Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
 
    predicted_as_id = 0
    seen = 0.
    abstained = 0
    # Show progress on progress bar
    for t in range(4):
        test_iter = tqdm(enumerate(test_loader[t]),
                            total=len(test_loader[t]),
                            desc=progress_prefix+' %s'%epoch,
                            mininterval=5.,
                            ncols=0)
        (xs, ys) = next(iter(test_loader[t]))
        # Iterate through the test set
        for i, (xs, ys) in test_iter:
            xs, ys = xs.to(device), ys.to(device)
            
            with torch.no_grad():
                # Use the model to classify this batch of input data
                _, pooled, out = net(xs, inference=True)
                max_out_score, ys_pred = torch.max(out, dim=1)
                ys_pred = torch.argmax(out, dim=1)
                abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))
                for j in range(len(ys_pred)):
                    seen+=1.
                    if isinstance(threshold, dict):
                        thresholdj = threshold[ys_pred[j].item()]
                    elif isinstance(threshold, float): #overall threshold
                        thresholdj = threshold
                    else:
                        raise ValueError("provided threshold should be float or dict", type(threshold))
                    sample_out = out[j,:]
                    
                    if sample_out.max().item() >= thresholdj:
                        predicted_as_id += 1
                        
                del out
                del pooled
                del ys_pred
    print("Samples seen:", seen, "of which predicted as In-Distribution:", predicted_as_id, flush=True)
    print("PIP-Net abstained from a decision for", abstained.item(), "images", flush=True)
    return predicted_as_id/seen
