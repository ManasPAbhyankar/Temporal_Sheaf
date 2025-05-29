import numpy as np


import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm




# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from models.sheaf_transformers import GeneralSheafTransformerV2,DiagSheafTransformerV2, OrthSheafTransformerV2,TrivialSheafTransformerV2
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
        
        #print(train_acc,tmp_test_acc)

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



def get_checkpoint(args):
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
    elif args["model"] == 'TrivialSheafTransformer':
        model_cls = TrivialSheafTransformerV2
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
    args["device"] = 'cuda:0'
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
        print(test_acc)
        # results.append([test_acc, best_val_acc])
        savename = f"oversmooth_metric_40_layer{args['model']}_{args['dataset']}_Fold_{fold}_d_{args['d']}.pth"
        torch.save(returned_model.state_dict(),os.path.join("/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/metric_checks",savename))

        
        break
    return savename

def input_conv(x):
    # x is embedding of all nodes in the graph... of shape (num_nodes,embedding_dim)

    # print(x)
    # print(x.shape)

    lfc = (1/x.shape[0])*(np.ones((x.shape[0],x.shape[0]))@x)
    hfc = x - lfc

    return np.linalg.norm(hfc,ord = 2)/np.linalg.norm(lfc,ord = 2)

def angle_conv(x):

    normed_x = x/np.linalg.norm(x,axis = 1,keepdims = True)

    tril_sims = np.tril(normed_x@normed_x.T)

    return (2/(x.shape[0]*(x.shape[0]-1)))*np.sum(tril_sims)

def rank_collapse(x):

    singular_vals = np.linalg.svd(x,compute_uv = False)
    #print(singular_vals)
    #singular_vals = singular_vals[np.nonzero(singular_vals)]
    #print(singular_vals)
    

    normed_vals = singular_vals/np.sum(singular_vals)
    normed_vals = normed_vals[np.nonzero(normed_vals)]
    

    return np.exp(np.sum(-normed_vals*np.log(normed_vals)))


def get_metrics(args,model_path):
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
    elif args["model"] == 'TrivialSheafTransformer':
        model_cls = TrivialSheafTransformerV2
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
    args["device"] = 'cuda:0'
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


    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], 0)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])


    model.load_state_dict(torch.load(model_path,map_location = args["device"]))


    ##### Creating Hooks for intermediate outputs ###########################################


    hook_array = []

    int_outputs = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input_arr, output_arr):
            #print("Input shape {name}".format(name = name),input_arr.detach().cpu().numpy().shape)
            # print("Output Shape {name}".format(name = name),output_arr.shape)
            int_outputs[name] = output_arr.clone().detach().cpu().numpy()
        return hook
    
    for layer in range(model.num_attention_layers):
        hook_array += [model.attention_layers[layer].register_forward_hook(getActivation(layer))]
    
    # hook_array += [model.lin2.register_forward_hook(getActivation(model.num_attention_layers))]


    model.eval()
    with torch.no_grad():
        logits = model(data.x)

        # print("LOGITSSSSS:::",logits.shape)
    
    # print("Output of last layer: ",int_outputs[model.num_attention_layers].shape)
    # print("Output of Second last layer: ",int_outputs[model.num_attention_layers-1].shape)
    
    # logits = logits.detach().cpu().numpy()
    
    
    
    for layer in range(model.num_attention_layers):
        #print(int_outputs[layer].shape)
        d = args['d']
        if model.add_lp:
            d+=1
        if model.add_hp:
            d+=1
        int_outputs[layer] = int_outputs[layer].reshape(-1,args['hidden_channels']*d)
    
    # print("Output of last layer: ",int_outputs[model.num_attention_layers].shape)
    # print("Output of Second last layer: ",int_outputs[model.num_attention_layers-1].shape)

    
    
    #print(int_outputs)




    ##########################################################################################

    

    input_conv_vals = []
    angle_conv_vals = []
    rank_collapse_vals = []

    for layer in range(model.num_attention_layers):
        #print(int_outputs[layer].shape)
        input_conv_vals += [input_conv(int_outputs[layer])]
        angle_conv_vals += [angle_conv(int_outputs[layer])]
        rank_collapse_vals += [rank_collapse(int_outputs[layer])]
    
    # input_conv_vals += [input_conv(logits)]
    # angle_conv_vals += [angle_conv(logits)]
    # rank_collapse_vals += [rank_collapse(logits)]

    for layer in range(model.num_attention_layers):
        hook_array[layer].remove()
    
    return input_conv_vals,angle_conv_vals,rank_collapse_vals



    

    

    
    


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
    'add_hp':True,
    'add_lp':True,
    'd':5,
    'dataset':'texas', 
    'dropout':0.1,
    'early_stopping':200,
    'epochs':1000,
    'folds':10 ,
    'hidden_channels':32,
    'input_dropout':0.1,
    'layers':2,
    'num_attention_layers':40,
    'num_heads':1,
    'lr':0.02 ,
    'model':"TrivialSheafTransformer",
    'second_linear':False,
    'sheaf_decay':0.0001, 
    'weight_decay':0.0001, 
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
    'use_residual':1
    }

    args['sheaf_decay'] = float(np.exp(scipy.stats.uniform.rvs(-11.0,4.2)))
    args['weight_decay'] = float(np.exp(scipy.stats.uniform.rvs(-9.2,2.4)))

    #ray.init(runtime_env={"working_dir": "/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master"},num_gpus = 2,num_cpus=64)

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

    # METRIC_TO_SEARCH = 'test_accuracy'
    
    # hyperopt_search = tune.search.hyperopt.HyperOptSearch(metric=METRIC_TO_SEARCH, mode="max",)
    # hyperopt_search = HyperOptSearch(n_initial_points=100)
    
    # tuner=  tune.Tuner(
    #     tune.with_resources(run_folds,{"gpu" : 0.2}),
    #     param_space=args,
    #     tune_config=tune.TuneConfig(num_samples=10,max_concurrent_trials = 10,metric = METRIC_TO_SEARCH,mode = "max",search_alg = hyperopt_search),
    #     run_config = raytrain.RunConfig(failure_config = raytrain.FailureConfig(max_failures = 10),
    #                                     storage_path = "/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/results",
    #                                     checkpoint_config = raytrain.CheckpointConfig(num_to_keep= 1, checkpoint_score_attribute = METRIC_TO_SEARCH, checkpoint_score_order = "max"))
    # )

    # tuner=  tune.Tuner(
    #     tune.with_resources(run_folds,{"gpu" : 0.2}),
    #     param_space=args,
    #     tune_config=tune.TuneConfig(num_samples=1000,max_concurrent_trials = 10,metric = METRIC_TO_SEARCH,mode = "max",search_alg = hyperopt_search),
    #     run_config = raytrain.RunConfig(failure_config = raytrain.FailureConfig(max_failures = 10),
    #                                     storage_path = "/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/results")
    # )

    # result = tuner.fit()
    # best_trial = result.get_best_result()

    # print(f"Best trial config: {best_trial.config}")
    # print(f"Validation Accuracy: {best_trial.metrics['accuracy']}")
    # print(f"Test Accuracy: {best_trial.metrics['test_accuracy']} +/- {best_trial.metrics['test_std']}")
    # print(f"Checkpoint Path = {best_trial.checkpoint}")
    

    # get_test_accuracy(best_trial.config)
    # get_test_accuracy(args)

    # get_checkpoint(args)

    # inputconv,angleconv,rankcollapse = get_metrics(model_path = os.path.join("/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/metric_checks",f"oversmooth_metric_{args['model']}_{args['dataset']}_Fold_0_d_{args['d']}.pth"),args = args)

    # print(inputconv)
    # print(angleconv)
    # print(rankcollapse)

    x_axis_1 = list(np.arange(1,args['num_attention_layers']+1))
    x_axis_2 = list(np.arange(1,args['num_attention_layers']+1))
    x_axis_3 = list(np.arange(1,args['num_attention_layers']+1))


    args['model'] = "DiagSheafTransformer"


    savedname = get_checkpoint(args)

    inputconv_diag,angleconv_diag,rank_coll_diag = get_metrics(model_path = os.path.join("/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/metric_checks",savedname),args = args)



    args['model'] = "OrthSheafTransformer"

    savedname = get_checkpoint(args)

    inputconv_orth,angleconv_orth,rank_coll_orth = get_metrics(model_path = os.path.join("/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/metric_checks",savedname),args = args)



    args['model'] = "TrivialSheafTransformer"
    args['d'] = 1

    savedname = get_checkpoint(args)

    inputconv_triv,angleconv_triv,rank_coll_triv = get_metrics(model_path = os.path.join("/home/manas/Temporal_Sheaf/neural-sheaf-diffusion-master/exp/metric_checks",savedname),args = args)




    # inputconv_diag = [0.6680032317885671, 1.2582723771569873, 1.859621603548378, 2.7133450348844304, 6.402465344152656, 9.800575801266351, 5.419801333923247, 5.489793307812655]
    # angleconv_diag = [1.0034373498768991, 1.008220999031706, 0.9997961609770012, 0.934720494242779, 0.9426880432579715, 0.9524751311700594, 0.8239984912628355, 0.9481290673977661]
    # rank_coll_diag = [2.4551814, 1.4249358, 1.6939919, 1.950109, 1.6814116, 1.127842, 1.8005307, 1.5609915]


    # inputconv_orth = [0.5258744922878823, 0.7647253384028737, 1.1971635673734968, 1.444315745467959, 2.2027770846393104, 4.199214557587415, 4.2556236208043625, 4.33006307128503]
    # angleconv_orth = [1.0101419883054104, 0.9756965946451089, 0.9242349234184531, 0.8552531239679037, 0.8724247437586321, 0.7713149587349126, 0.9368327731865129, 1.003523436092596]
    # rank_coll_orth = [1.5129709, 2.1120024, 2.337172, 1.9416633, 1.8294713, 2.817447, 2.1076267, 1.43414]

    # inputconv_triv = [0.7728745184961098, 0.49094840789498967, 0.5636296104405129, 1.3569107369170268, 1.157518127684733, 0.7666909250508539, 0.8918800862198729, 0.933954021155245]
    # angleconv_triv = [0.8882561630224284, 1.0101198216912868, 0.9112119203371765, 0.7260288592896175, 0.46883362325857203, 0.9385605376095899, 0.6515052890245001, 0.6422344251974119]
    # rank_coll_triv = [2.1072507, 1.26384, 3.012415, 2.5082812, 2.9852178, 3.4214773, 5.3455505, 4.707684]


    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize = (18,6),nrows=1, ncols=3)

    ax[0].plot(x_axis_1,inputconv_diag, color = 'blue',label = "DiagSheafTransf (d = 5)")
    ax[0].plot(x_axis_1,inputconv_orth , color = 'green',label = "OrthSheafTransf (d = 5)")
    ax[0].plot(x_axis_1,inputconv_triv , color = 'red',label = "TrivialSheafTransf")
    ax[0].set_xlabel("Layer")
    ax[0].set_ylabel("ICM")
    ax[0].set_title("Input Convergence")
    ax[0].legend()

    ax[1].plot(x_axis_2,angleconv_diag, color = 'blue',label = "DiagSheafTransf (d = 5)")
    ax[1].plot(x_axis_2,angleconv_orth , color = 'green',label = "OrthSheafTransf (d = 5)")
    ax[1].plot(x_axis_2,angleconv_triv , color = 'red',label = "TrivialSheafTransf")
    ax[1].set_xlabel("Layer")
    ax[1].set_ylabel("ACM")
    ax[1].set_title("Angle Convergence")
    ax[1].legend()

    ax[2].plot(x_axis_3,rank_coll_diag, color = 'blue',label = "DiagSheafTransf (d = 5)")
    ax[2].plot(x_axis_3,rank_coll_orth , color = 'green',label = "OrthSheafTransf (d = 5)")
    ax[2].plot(x_axis_3,rank_coll_triv , color = 'red',label = "TrivialSheafTransf")
    ax[2].set_xlabel("Layer")
    ax[2].set_ylabel("Rank Collapse Metric")
    ax[2].set_title("Rank Collapse")
    ax[2].legend()






    # plt.plot(x_axis,y_axis,label = "d = 1",color = 'red')
    # plt.xlabel("Layer")
    # plt.ylabel("ICM")

    # plt.legend()

    # plt.savefig(f"{args['dataset']}_DATASET_40_LAYERS.png")
    plt.savefig(f"{args['dataset']}_DATASET_40_Layer.png")



    """    oversmoothing metrics:

    Diagsheaf-Texas

    [0.6541164282469893, 1.0572595488102123, 1.3067066214447556, 1.2167508937898013, 1.6016053389269511, 2.817792528027615, 3.222945465445651,3.8548459502390444]
    [0.9363637557797394, 0.9893581486819192, 0.9793647587897075, 0.9990041443358554, 0.9193711108734162, 0.8964226251463701, 0.997769381830601, 1.0089577750630516]
    [2.5888352, 1.7224656, 1.2816565, 1.0810187, 1.2958332, 2.2340806, 1.2938168, 1.0495708]


    DiagSheaf-Wisconsin

    [0.42943465678496695, 1.3082708756408468, 2.394128874798023, 5.239457316017578, 5.816689556702949, 4.290287877898269, 3.8894421209105774, 5.17288842937083]
    [0.9920737051792828, 1.0070283864541834, 1.007699389940239, 0.9854970742031872, 0.6576137948207171, 0.8406984561752988, 0.9843156125498008, 1.00550031125498]
    [2.0534415, 1.2108852, 1.0562336, 1.2429539, 1.2427334, 1.1249362, 1.067547, 1.1261576]


    Diagsheaf Cornell

    [0.6680032317885671, 1.2582723771569873, 1.859621603548378, 2.7133450348844304, 6.402465344152656, 9.800575801266351, 5.419801333923247, 5.489793307812655]
    [1.0034373498768991, 1.008220999031706, 0.9997961609770012, 0.934720494242779, 0.9426880432579715, 0.9524751311700594, 0.8239984912628355, 0.9481290673977661]
    [2.4551814, 1.4249358, 1.6939919, 1.950109, 1.6814116, 1.127842, 1.8005307, 1.5609915]








    Orthsheaf-Texas

    [0.5251113478038477, 1.2932417195080785, 2.3643930395351127, 3.0221464219623706, 3.987608517202869, 5.947468459669402, 2.8710704258507787, 3.307580244935124]
    [0.986315458363358, 0.931426582297484, 0.9022328582875457, 0.7170805842040473, 0.6047079894576053, 0.37600834637527775, 0.874120665664595, 1.0047726244895814]
    [2.2279727, 3.6609256, 2.7622535, 1.805844, 4.4694324, 7.1522927, 5.1993093, 1.8419034]


    Orthsheaf - Wisconsin

    [0.5768138155114557, 0.7630737435234659, 0.9274623611855006, 2.885424336695044, 2.782524276627411, 4.08267925467123, 3.4036087329355786, 3.8310890702073377]
    [0.872159985059761, 0.919801045816733, 0.8666688247011952, 0.44876967131474105, 0.2905980141932271, 0.10089250809262948, 0.9357094123505976, 1.0070288844621513]
    [3.4392505, 3.8767195, 4.5582213, 7.3105893, 11.878811, 18.252832, 4.9015074, 1.3735043]

    Orthsheaf - Cornell

    [0.5258744922878823, 0.7647253384028737, 1.1971635673734968, 1.444315745467959, 2.2027770846393104, 4.199214557587415, 4.2556236208043625, 4.33006307128503]
    [1.0101419883054104, 0.9756965946451089, 0.9242349234184531, 0.8552531239679037, 0.8724247437586321, 0.7713149587349126, 0.9368327731865129, 1.003523436092596]
    [1.5129709, 2.1120024, 2.337172, 1.9416633, 1.8294713, 2.817447, 2.1076267, 1.43414]




    TrivialSheaf - Texas

    [1.1127986010560738, 0.9894146856233385, 0.5813028409820984, 0.8914390151770979, 0.8364529090312812, 0.5961812775080587, 1.9857519085595623, 2.2108218353891504]
    [0.49178073090246505, 0.8372218148569327, 1.0093891444109169, 0.8895163174428031, 0.8048555675028524, 0.8807376345478293, 0.13302478344742688, 0.23487962766187023]
    [3.6798809, 1.8626117, 1.3340279, 2.858351, 4.0110664, 5.7700586, 5.43139, 2.9951203]




    TrivialSheaf-Wisconsin


    [2.366320583456794, 1.1052188820973115, 2.11740905946661, 2.2104707039418727, 1.29350875279819, 0.9275130285383386, 2.142923233486979, 3.0218955645811385]
    [0.09302339081175298, 0.4076755789342629, 0.2262476655876494, 0.21699814803286852, 0.4765148779880478, 0.7243905627490039, 0.2248449950199203, 0.10604315550298805]
    [1.3460686, 3.5099733, 3.9446208, 5.1635494, 4.896316, 5.452895, 3.7759945, 3.0339348]


    TrivialSheaf Cornell

    [0.7728745184961098, 0.49094840789498967, 0.5636296104405129, 1.3569107369170268, 1.157518127684733, 0.7666909250508539, 0.8918800862198729, 0.933954021155245]
    [0.8882561630224284, 1.0101198216912868, 0.9112119203371765, 0.7260288592896175, 0.46883362325857203, 0.9385605376095899, 0.6515052890245001, 0.6422344251974119]
    [2.1072507, 1.26384, 3.012415, 2.5082812, 2.9852178, 3.4214773, 5.3455505, 4.707684]"""

    


    






    #pass
    
    
    
"""

TODO: Forward hook not working properly .... Dimension of output of forward hook of last layer is not output_channels....

"""




