from nni.algorithms.compression.pytorch.pruning import LevelPruner, SlimPruner, L2FilterPruner, \
    L1FilterPruner


def get_pruner(pruner_name, model, sparsity=0.5):
    if pruner_name == "l1unstructure":
        config_list = [{
            'sparsity': sparsity,
            'op_types': ["default"]
        }]
        return LevelPruner(model, config_list)
    elif pruner_name == "slim":
        config_list = [{
            'sparsity': sparsity,
            'op_types': ["BatchNorm2d"]
        }]
        return SlimPruner(model, config_list)
    elif pruner_name == "l1structure":
        config_list = [{
            'sparsity': sparsity,
            'op_types': ['Conv2d']
        }]
        return L1FilterPruner(model, config_list)
    elif pruner_name == "l2structure":
        config_list = [{
            'sparsity': sparsity,
            'op_types': ['Conv2d']
        }]
        return L2FilterPruner(model, config_list)
    else:
        raise ValueError
