import torch
import random
import numpy as np
from copy import deepcopy


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    teacher_logits = None
    teacher_divergences = None
    max_len = max([len(f["input_ids"]) for (f, _) in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for (f, _) in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for (f, _) in batch]
    labels = [f["labels"] for (f, _) in batch]
    entity_pos = [f["entity_pos"] for (f, _) in batch]
    hts = [f["hts"] for (f, _) in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    if "teacher_logits" in batch[0][0]:
        teacher_logits = [f["teacher_logits"] for (f, _) in batch]
        teacher_divergences = [f["teacher_divergences"] for (f, _) in batch]
    input_indices = [idx for (_, idx) in batch]
    output = (input_indices, input_ids, input_mask, labels, entity_pos, hts, teacher_logits, teacher_divergences)
    return output


def add_logits_to_features(features, indices, logits, divergences):
    for i, idx in enumerate(indices):
        feature = features[idx][0]
        if 'teacher_logits' in feature:
            feature.pop('teacher_logits')
            feature.pop('teacher_divergences')
        feature['teacher_logits'] = logits[i]
        feature['teacher_divergences'] = divergences[i]
        assert logits[i].shape[0] == len(feature['hts'])
