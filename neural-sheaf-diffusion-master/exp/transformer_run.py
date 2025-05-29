import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm

import ray
from ray import tune
from ray import train as raytrain
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import FIFOScheduler
import ray.cloudpickle as pickle
import ray.train.torch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch




# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from models.sheaf_transformers import GeneralSheafTransformerV2,DiagSheafTransformerV2, OrthSheafTransformerV2
from utils.heterophilic import get_dataset, get_fixed_splits





import tempfile


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)
        # if fold == 0:
        #     # res_dict = {
        #     #     f'fold{fold}_train_acc': train_acc,
        #     #     f'fold{fold}_train_loss': train_loss,
        #     #     f'fold{fold}_val_acc': val_acc,
        #     #     f'fold{fold}_val_loss': val_loss,
        #     #     f'fold{fold}_tmp_test_acc': tmp_test_acc,
        #     #     f'fold{fold}_tmp_test_loss': tmp_test_loss,
        #     # }
        #     #wandb.log(res_dict, step=epoch)

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
        
        # if "ODE" not in args['model']:
        # # Debugging for discrete models
        #     for i in range(len(model.attention_layers)):
        #         # L_max = model.sheaf_learners[i].L.detach().max().item()
        #         # L_min = model.sheaf_learners[i].L.detach().min().item()
        #         # L_avg = model.sheaf_learners[i].L.detach().mean().item()
        #         # L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
        #         # print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")
        #         weight_max = model.sheaf_learners[i].linear1.weight.clone().detach().max().item()
        #         weight_min = model.sheaf_learners[i].linear1.weight.clone().detach().min().item()
        #         print(f"Laplacian {i}: Weight Max: {weight_max:.4f}, Weight_Min: {weight_min:.4f}")
                

        if bad_counter == args['early_stopping']:
            break

    # print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    # print(f"Test acc: {test_acc:.4f}")
    # print(f"Best val acc: {best_val_acc:.4f}")

    # if "ODE" not in args['model']:
    #     # Debugging for discrete models
    #     for i in range(len(model.attention_layers)):
    #         L_max = model.sheaf_learners[i].L.detach().max().item()
    #         L_min = model.sheaf_learners[i].L.detach().min().item()
    #         L_avg = model.sheaf_learners[i].L.detach().mean().item()
    #         L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
    #         print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        # with np.printoptions(precision=3, suppress=True):
        #     for i in range(0, args['layers']):
        #         print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    #wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, keep_running , best_epoch


def get_validation_accuracy(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc =  0
    best_val_loss = float('inf')
    best_epoch = 0
    bad_counter = 0
    test_acc = -1

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    # print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    # print(f"Test acc: {test_acc:.4f}")
    # print(f"Best val acc: {best_val_acc:.4f}")

    # if "ODE" not in args['model']:
    #     # Debugging for discrete models
    #     for i in range(len(model.sheaf_learners)):
    #         L_max = model.sheaf_learners[i].L.detach().max().item()
    #         L_min = model.sheaf_learners[i].L.detach().min().item()
    #         L_avg = model.sheaf_learners[i].L.detach().mean().item()
    #         L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            #print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        # with np.printoptions(precision=3, suppress=True):
        #     for i in range(0, args['layers']):
        #         print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    #wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    # return test_acc,best_val_acc, keep_running
    return model ,test_acc,best_val_acc, keep_running



def run_folds(args):
    """TODO: Add Raytune hyperopt lines
       TODO: Add best model returner in folds code"""
    
    #print("###################STARTED FOLDS !!!!!!!!!!!!!!!!!!###########################")
    
    
    
    
    # args = {
    # 'add_hp':False,
    # 'add_lp':False,
    # 'd':4 ,
    # 'dataset':'texas', 
    # 'dropout':0 ,
    # 'early_stopping':100, 
    # 'epochs':10 ,
    # 'folds':10 ,
    # 'hidden_channels':32, 
    # 'input_dropout':0.7 ,
    # 'layers':2,
    # 'num_attention_layers':5,
    # 'num_heads':1,
    # 'lr':0.01 ,
    # 'model':"GeneralSheafTransformer", 
    # 'second_linear':False,
    # 'sheaf_decay':0.0012638885974822734, 
    # 'weight_decay':0.0002969905682317406, 
    # 'left_weights':True ,
    # 'right_weights':True ,
    # 'use_act':True ,
    # 'sheaf_act':"tanh",
    # 'normalised':True, 
    # 'edge_weights':True, 
    # 'stop_strategy':'acc',
    # 'deg_normalised':False,
    # 'linear':False,
    # 'orth':'householder',
    # 'sparse_learner':False,
    # 'seed':43,
    # 'cuda':0,
    # 'evectors':0,
    # 'entity':None,
    # 'min_acc':0.0,
    # }
    #print(args['dataset'])
    #print(args.dataset)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    


    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha

    if args["model"] == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args["model"] == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args["model"]== 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    elif args["model"] == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args["model"] == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args["model"] == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    elif args["model"] == 'GeneralSheafTransformer':
        model_cls = GeneralSheafTransformer
    elif args["model"] == 'DiagSheafTransformer':
        model_cls = DiagSheafTransformerV2
    elif args["model"] == 'OrthSheafTransformer':
        model_cls = OrthSheafTransformerV2
    else:
        raise ValueError(f'Unknown model {args["model"]}')

    # model_cls = GeneralSheafTransformer

    dataset = get_dataset(args["dataset"])
    if args["evectors"] > 0:
        dataset = append_top_k_evectors(dataset, args["evectors"])

    # Add extra arguments
    #args.sha = sha
    args["graph_size"] = dataset[0].x.size(0)
    args["input_dim"] = dataset.num_features
    args["output_dim"] = dataset.num_classes
    #args["device"] = torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu')
    args["device"] = ray.train.torch.get_device()
    assert args["normalised"] or args["deg_normalised"]
    # if args.sheaf_decay is None:
    #     args.sheaf_decay = args.weight_decay
    
    if "sheaf_decay" not in args.keys():
        args["sheaf_decay"] = args["weight_decay"]

    # Set the seed for everything
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    #raytrain.torch.enable_reproducibility(seed = args["seed"])

    results = []
    #print(f"Running with wandb account: {args['entity']}")
    # print(args)
    #wandb.init(project="sheaf", config=vars(args), entity=args["entity"])
    
    
    """------------------------------------With saving checkpoints"""
    
    # with tempfile.TemporaryDirectory() as tmpdir:
        

    #     for fold in tqdm(range(args["folds"])):
    #         returned_model,test_acc, best_val_acc, keep_running = get_validation_accuracy(args, dataset, model_cls, fold)
    #         results.append([test_acc, best_val_acc])
            
    #         torch.save(returned_model.state_dict,os.path.join(tmpdir,f"{args['model']}_{args['dataset']}_Fold_{fold}"))
    #         if not keep_running:
    #             break

    #     test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    #     test_std = np.sqrt(np.var(results,axis=0)[0]) *100

    #     #wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    #     #wandb.log(wandb_results)
    #     #wandb.finish()

    #     #model_name = args["model"] if args["evectors"] == 0 else f"{args['model']}+LP{args['evectors']}"
    #     #print(f'{model_name} on {args.dataset} | SHA: {sha}')
    #     #print(f'Val acc: {val_acc_mean:.4f}')
    #     #return val_acc_mean #Get this back to Ray Tune

    #     #raytrain.report(accuracy= val_acc_mean)
    #     checkpoint = Checkpoint.from_directory(tmpdir)
    #     # return {"accuracy": float(val_acc_mean),"test_accuracy":float(test_acc_mean),"test_std":float(test_std)}
    #     metrics = {"accuracy": float(val_acc_mean),"test_accuracy":float(test_acc_mean),"test_std":float(test_std)}
    #     raytrain.report(metrics, checkpoint = checkpoint)
    
    
    """------------------------------------Without saving checkpoints"""
    
    for fold in tqdm(range(args["folds"])):
        returned_model,test_acc, best_val_acc, keep_running = get_validation_accuracy(args, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        
        if not keep_running:
            break

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_std = np.sqrt(np.var(results,axis=0)[0]) *100

    #wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    #wandb.log(wandb_results)
    #wandb.finish()

    #model_name = args["model"] if args["evectors"] == 0 else f"{args['model']}+LP{args['evectors']}"
    #print(f'{model_name} on {args.dataset} | SHA: {sha}')
    #print(f'Val acc: {val_acc_mean:.4f}')
    #return val_acc_mean #Get this back to Ray Tune

    #raytrain.report(accuracy= val_acc_mean)
    # return {"accuracy": float(val_acc_mean),"test_accuracy":float(test_acc_mean),"test_std":float(test_std)}
    metrics = {"accuracy": float(val_acc_mean),"test_accuracy":float(test_acc_mean),"test_std":float(test_std)}
    raytrain.report(metrics)


def get_test_accuracy(args):
    """TODO: Add Raytune hyperopt lines
       TODO: Add best model returner in folds code"""
       
    #tune.utils.wait_for_gpu()
    
    
    
    
    # args = {
    # 'add_hp':False,
    # 'add_lp':False,
    # 'd':4 ,
    # 'dataset':'texas', 
    # 'dropout':0 ,
    # 'early_stopping':100, 
    # 'epochs':10 ,
    # 'folds':10 ,
    # 'hidden_channels':32, 
    # 'input_dropout':0.7 ,
    # 'layers':2,
    # 'num_attention_layers':5,
    # 'num_heads':1,
    # 'lr':0.01 ,
    # 'model':"GeneralSheafTransformer", 
    # 'second_linear':False,
    # 'sheaf_decay':0.0012638885974822734, 
    # 'weight_decay':0.0002969905682317406, 
    # 'left_weights':True ,
    # 'right_weights':True ,
    # 'use_act':True ,
    # 'sheaf_act':"tanh",
    # 'normalised':True, 
    # 'edge_weights':True, 
    # 'stop_strategy':'acc',
    # 'deg_normalised':False,
    # 'linear':False,
    # 'orth':'householder',
    # 'sparse_learner':False,
    # 'seed':43,
    # 'cuda':0,
    # 'evectors':0,
    # 'entity':None,
    # 'min_acc':0.0,
    # }
    #print(args['dataset'])
    #print(args.dataset)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    


    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha

    if args["model"] == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args["model"] == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args["model"]== 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    elif args["model"] == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args["model"] == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args["model"] == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    elif args["model"] == 'GeneralSheafTransformer':
        model_cls = GeneralSheafTransformerV2
    elif args["model"] == 'DiagSheafTransformer':
        model_cls = DiagSheafTransformerV2
    elif args["model"] == 'OrthSheafTransformer':
        model_cls = OrthSheafTransformerV2
    else:
        raise ValueError(f'Unknown model {args["model"]}')

    # model_cls = GeneralSheafTransformer

    dataset = get_dataset(args["dataset"])
    if args["evectors"] > 0:
        dataset = append_top_k_evectors(dataset, args["evectors"])

    # Add extra arguments
    #args.sha = sha
    args["graph_size"] = dataset[0].x.size(0)
    args["input_dim"] = dataset.num_features
    args["output_dim"] = dataset.num_classes
    args["device"] = torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu')
    assert args["normalised"] or args["deg_normalised"]
    # if args.sheaf_decay is None:
    #     args.sheaf_decay = args.weight_decay
    
    if "sheaf_decay" not in args.keys():
        args["sheaf_decay"] = args["weight_decay"]

    # Set the seed for everything
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    #raytrain.torch.enable_reproducibility(seed = args["seed"])



    results = []
    #print(f"Running with wandb account: {args['entity']}")
    #print(args)
    #wandb.init(project="sheaf", config=vars(args), entity=args["entity"])

    for fold in tqdm(range(args["folds"])):
        test_acc, best_val_acc, keep_running, best_epoch = run_exp(args,dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        if not keep_running:
            break

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    #wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    #wandb.log(wandb_results)
    #wandb.finish()

    #model_name = args["model"] if args["evectors"] == 0 else f"{args['model']}+LP{args['evectors']}"
    print(f'{args["model"]} on {args["dataset"]}')
    print(f'Configuration tested : {args}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
    #return val_acc_mean

    #train.report({"accuracy": val_acc_mean})


import scipy

if __name__ == '__main__':
    # parser = get_parser()
    # args = parser.parse_args()
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)


    # args = {'add_hp': True, 'add_lp': True, 'd': 7, 'dataset': 'texas', 'dropout': 0.4, 'early_stopping': 500, 'epochs': 1, 'folds': 10, 'hidden_channels': 8, 'input_dropout': 0.1, 'layers': 2, 'num_attention_layers': 2, 'num_heads': 4, 'lr': 0.02, 'model': 'DiagSheafTransformer', 'second_linear': False, 'sheaf_decay': 0.0008446800447178244, 'weight_decay': 0.0006928530011451836, 'left_weights': True, 'right_weights': True, 'use_act': True, 'sheaf_act': 'tanh', 'normalised': True, 'edge_weights': True, 'stop_strategy': 'acc', 'deg_normalised': False, 'linear': False, 'orth': 'householder', 'sparse_learner': False, 'seed': 43, 'cuda': 0, 'evectors': 0, 'entity': None, 'min_acc': 0.0,'use_residual': 0,'graph_size': 183, 'input_dim': 1703, 'output_dim': 5}

    # args = {
    # 'add_hp':False,
    # 'add_lp':False,
    # 'd':tune.grid_search([2,3,4,5]),
    # 'dataset':'texas', 
    # 'dropout':tune.uniform(0,0.9),
    # 'early_stopping':20,
    # 'epochs':50 ,
    # 'folds':10 ,
    # 'hidden_channels':tune.grid_search([8,16,32]),
    # 'input_dropout':tune.uniform(0,0.9),
    # 'layers':2,
    # 'num_attention_layers':tune.grid_search([2,3,4,5,6,7,8]),
    # 'num_heads':tune.grid_search([1,2,4,8,16]),
    # 'lr':tune.grid_search([0.01,0.02]) ,
    # 'model':"DiagSheafTransformer",
    # 'second_linear':False,
    # 'sheaf_decay':tune.sample_from(lambda _:float(np.exp(scipy.stats.uniform.rvs(-11.0,4.2)))), 
    # 'weight_decay':tune.sample_from(lambda _:float(np.exp(scipy.stats.uniform.rvs(-9.2,2.4)))), 
    # 'left_weights':True ,
    # 'right_weights':True ,
    # 'use_act':True ,
    # 'sheaf_act':"tanh",
    # 'normalised':True, 
    # 'edge_weights':True,
    # 'stop_strategy':'acc',
    # 'deg_normalised':False,
    # 'linear':False,
    # 'orth':'householder',
    # 'sparse_learner':False,
    # 'seed':43,
    # 'cuda':0,
    # 'evectors':0,
    # 'entity':None,
    # 'min_acc':0.0,
    # 'use_residual':tune.grid_search([1,0])
    # }

    args = {
    'add_hp':tune.choice([True,False]),
    'add_lp':tune.choice([True,False]),
    'd':tune.choice([2,3,4,5,6]),   # 2-7
    'dataset':'citeseer', 
    'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
    'early_stopping':200,
    'epochs':1000,
    'folds':10 ,
    'hidden_channels':tune.choice([8,16,32,64]),
    'input_dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
    'layers':2,
    'num_attention_layers':tune.choice([2,3,4,5,6,7,8]),
    'num_heads':tune.choice([1,2,4,8,16]),
    'lr':0.02 ,
    'model':"OrthSheafTransformer",
    'second_linear':False,
    'sheaf_decay':tune.sample_from(lambda _:float(np.exp(scipy.stats.uniform.rvs(-11.0,4.2)))), 
    'weight_decay':tune.sample_from(lambda _:float(np.exp(scipy.stats.uniform.rvs(-9.2,2.4)))), 
    'left_weights':True ,
    'right_weights':True ,
    'use_act':True ,
    'sheaf_act':"tanh",
    'normalised':True, 
    'edge_weights':True,
    'stop_strategy':'acc',
    'deg_normalised':False,
    'linear':False,
    'orth':'householder',
    'sparse_learner':False,
    'seed':43,
    'cuda':0,
    'evectors':0,
    'entity':None,
    'min_acc':0.0,
    'use_residual':tune.choice([1,0])
    }
    ray.init(runtime_env={"working_dir": "/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master"},num_gpus = 2,num_cpus=64)

    # scheduler = FIFOScheduler()
    # result = tune.run(
    #     run_folds,
    #     resources_per_trial={"cpu": 1, "gpu": 1},
    #     config=args,
    #     num_samples=1,
    # )
    

    # tuner=  tune.Tuner(
    #     tune.with_resources(run_folds,{"gpu" : 0.25}),
    #     param_space=args,
    #     tune_config=tune.TuneConfig(num_samples= 8,max_concurrent_trials = 15,metric = "accuracy",mode = "max"),
    #     run_config = raytrain.RunConfig(failure_config = raytrain.FailureConfig(max_failures = 10),
    #                                     checkpoint_config = raytrain.CheckpointConfig(num_to_keep= 1, checkpoint_score_attribute = "accuracy", checkpoint_score_order = "max"))
    # )

    METRIC_TO_SEARCH = 'accuracy'
    
    hyperopt_search = tune.search.hyperopt.HyperOptSearch(metric=METRIC_TO_SEARCH, mode="max",)
    hyperopt_search = HyperOptSearch(n_initial_points=300)
    
    # tuner=  tune.Tuner(
    #     tune.with_resources(run_folds,{"gpu" : 0.2}),
    #     param_space=args,
    #     tune_config=tune.TuneConfig(num_samples=10,max_concurrent_trials = 10,metric = METRIC_TO_SEARCH,mode = "max",search_alg = hyperopt_search),
    #     run_config = raytrain.RunConfig(failure_config = raytrain.FailureConfig(max_failures = 10),
    #                                     storage_path = "/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/results",
    #                                     checkpoint_config = raytrain.CheckpointConfig(num_to_keep= 1, checkpoint_score_attribute = METRIC_TO_SEARCH, checkpoint_score_order = "max"))
    # )

    tuner=  tune.Tuner(
        tune.with_resources(run_folds,{"gpu" : 0.5}),
        param_space=args,
        tune_config=tune.TuneConfig(num_samples=2000,max_concurrent_trials = 10,metric = METRIC_TO_SEARCH,mode = "max",search_alg = hyperopt_search),
        run_config = raytrain.RunConfig(failure_config = raytrain.FailureConfig(max_failures = 10),
                                        storage_path = "/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/results")
    )

    result = tuner.fit()
    best_trial = result.get_best_result()

    print(f"Best trial config: {best_trial.config}")
    print(f"Validation Accuracy: {best_trial.metrics['accuracy']}")
    print(f"Test Accuracy: {best_trial.metrics['test_accuracy']} +/- {best_trial.metrics['test_std']}")
    print(f"Checkpoint Path = {best_trial.checkpoint}")
    

    # get_test_accuracy(best_trial.config)
    #get_test_accuracy(args)


    






    #pass
    
    
    
"""

TODO: Add model checkpoints for all 10 folds for each iteration. Configure so as to save best checkpoint.

"""
