import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import ast
cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)

def extract_log_from_raw(path_results, path_exp, num_tasks = 4, dataset = "cub_200_2011_cropped_"):
    # folder_path = "./"+str(path_results)+"/cub_200_2011_cropped_icicle_"+str(path_exp)+"/"
    folder_path = "./"+str(path_results)+"/"+str(dataset)+str(path_exp)+"/"
    if not os.path.exists(folder_path):
        folder_path = "./"+str(path_results)+"/"+str(path_exp)+"/"
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    files = [f for f in files if os.path.isfile(f) if 'raw_log' in f]
    latest_rawlog = max(files, key=os.path.getmtime)
    d = {'train':{t:{} for t in range(num_tasks)},'test':{}, 'val':{t:{} for t in range(num_tasks)}, 'pretrain':{}}
    with open(latest_rawlog, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            dd = ast.literal_eval(line)
            dd_v = dd.values()
            if dd['group'] == 'train':
                if dd['name'] not in d[dd['group']][dd['task']]:
                    d[dd['group']][dd['task']][dd['name']] = [dd['value']]
                else:
                    d[dd['group']][dd['task']][dd['name']].append(dd['value'])
            elif dd['group'] == 'val':
                if dd['name'] not in d[dd['group']][dd['task']]:
                    d[dd['group']][dd['task']][dd['name']] = [dd['value']]
                else:
                    d[dd['group']][dd['task']][dd['name']].append(dd['value'])
            elif dd['group'] == 'pretrain':
                if dd['name'] not in d[dd['group']]:
                    d[dd['group']][dd['name']] = [dd['value']]
                else:
                    d[dd['group']][dd['name']].append(dd['value'])
            else:
                if dd['name'] not in d[dd['group']]:
                    d[dd['group']][dd['name']] = [dd['value']]
                else:
                    d[dd['group']][dd['name']].append(dd['value'])
    return d

def create_plots(d, d_proto, path_results, path_exp, dataset):
    # p = "./"+str(path_results)+"/cub_200_2011_cropped_icicle_"+str(path_exp)+"/results/"
    p = "./"+str(path_results)+"/"+str(dataset)+str(path_exp)+"/results/"
    if not os.path.exists(p): 
        p = "./"+str(path_results)+"/"+str(path_exp)+"/results/"
    # for task in d['train'].keys():
    #     for loss in d['train'][task].keys():
    #         plt.plot(d['train'][task][loss])
    #         plt.savefig(p+"task_"+str(task)+"_"+str(loss)+".png")
    #         plt.show()
    #         plt.clf()
    #         plt.cla()
    #         plt.close()
    for loss in d['pretrain'].keys():
        plt.plot(d['pretrain'][loss])
        plt.savefig(p+"pretrain_"+str(loss)+".png")
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
    for task in d['train'].keys():
        plt.plot(d['train'][task]['LA'], label='LA_'+str(task))
    plt.legend(loc="upper right", prop={'size':12})
    plt.savefig(p+"alltasks_LA.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    for task in d['train'].keys():
        plt.plot(d['train'][task]['LT'], label='LT'+str(task))
    plt.legend(loc="upper right", prop={'size':12})
    plt.savefig(p+"alltasks_LT.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    for task in d['train'].keys():
        plt.plot(d['train'][task]['LC'], label='LC'+str(task))
    plt.legend(loc="upper right", prop={'size':12})
    plt.savefig(p+"alltasks_LC.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    # for task in d['train'].keys():
    #     plt.plot(d['train'][task]['loss_p'], label='loss_p'+str(task))
    # plt.legend(loc="upper right", prop={'size':12})
    # plt.savefig(p+"alltasks_loss_p.png")
    # plt.show()
    # plt.clf()
    # plt.cla()
    # for task in d['train'].keys():
    #     plt.plot(d['train'][task]['loss_d'], label='loss_d'+str(task))
    # plt.legend(loc="upper right", prop={'size':12})
    # plt.savefig(p+"alltasks_loss_d.png")
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()
    # for task in d['train'].keys():
    #     plt.plot(d['train'][task]['loss_cw'], label='loss_cw'+str(task))
    # plt.legend(loc="upper right", prop={'size':12})
    # plt.savefig(p+"alltasks_loss_cw.png")
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()
    for task in d['train'].keys():
        plt.plot(d['train'][task]['loss_pr1'], label='loss_pr1'+str(task))
    plt.legend(loc="upper right", prop={'size':12})
    plt.savefig(p+"alltasks_loss_pr1.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    # for task in d['train'].keys():
    #     plt.plot(d['train'][task]['loss_pr2'], label='loss_pr2'+str(task))
    # plt.legend(loc="upper right", prop={'size':12})
    # plt.savefig(p+"alltasks_loss_pr2.png")
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()
    for task in d['train'].keys():
        plt.plot(d['train'][task]['loss_ho'], label='loss_ho'+str(task))
    plt.legend(loc="upper right", prop={'size':12})
    plt.savefig(p+"alltasks_loss_ho.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    for task in d['train'].keys():
        plt.plot(d['train'][task]['loss'], label='loss'+str(task))
    plt.legend(loc="upper right", prop={'size':12})
    plt.savefig(p+"alltasks_loss.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    for task in d['train'].keys():
        plt.plot(d['train'][task]['dec_loss'], label='dec_loss'+str(task))
    plt.legend(loc="upper right", prop={'size':12})
    plt.savefig(p+"alltasks_dec_loss.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    # print(d['val'])
    for task in d['val'].keys():
        plt.plot(d['val'][task]['loss'], label='loss'+str(task))
        plt.legend(loc="upper right", prop={'size':12})
        plt.savefig(p+"alltasks_VAL_loss"+str(task)+".png")
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    #plot hist for protos
    # p = "./"+str(path_results)+"/cub_200_2011_cropped_icicle_"+str(path_exp)+"/figures/"
    # if not os.path.exists(p): 
    #     p = "./"+str(path_results)+"/"+str(path_exp)+"/figures/"
    # for task in d_proto.keys():
    #     plt.cla()
    #     plt.clf()
    #     plt.bar(range(len(d_proto[task])),d_proto[task])
    #     plt.savefig(p+"/proto_hist_task_"+str(task)+".png")
    #     plt.show()
    plt.cla()
    plt.clf()
    plt.close()
    return

    