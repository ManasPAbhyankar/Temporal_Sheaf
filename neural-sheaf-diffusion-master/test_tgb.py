from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
name = 'tgbn-trade'
dataset = PyGNodePropPredDataset(name=name, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

eval_metric = dataset.eval_metric
num_classes = dataset.num_classes
data = dataset.get_TemporalData()

print(num_classes)
print(train_mask)
print(data.num_events)
print(data.num_nodes)
from tgb.nodeproppred.evaluate import Evaluator

evaluator = Evaluator(name=name)