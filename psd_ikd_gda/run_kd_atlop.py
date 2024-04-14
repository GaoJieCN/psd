import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
# from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn, add_logits_to_features
from prepro import read_docred, read_gda
from evaluation import to_official, official_evaluate
from niceutils import *
from tensorboardX import SummaryWriter


def print_features(features):
    for feature in features:
        print(feature['title'])
        print(*feature['input_ids'], sep=' ')
        if 'teacher_logits' in feature:
            for logits in feature['teacher_logits']:
                numpy_logits = logits.cpu().numpy()
                print(*numpy_logits, sep=' ')


def print_inputs(inputs):
    for idx, input_ids in enumerate(inputs['input_ids']):
        numpy_input_ids = input_ids.cpu().numpy()
        print(*numpy_input_ids, sep=' ')
        if inputs['teacher_logits'] is not None:
            for logits in inputs['teacher_logits'][idx]:
                numpy_logits = logits.cpu().numpy()
                print(*numpy_logits, sep=' ')


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        best_output = {}
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[1].to(args.device),
                          'attention_mask': batch[2].to(args.device),
                          'labels': batch[3],
                          'entity_pos': batch[4],
                          'hts': batch[5],
                          'teacher_logits': batch[6],
                          'current_step': num_steps,
                          'total_steps': total_steps,
                          'current_epoch': epoch,
                          'num_epoch': num_epoch,
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()

                input_indices = batch[0]
                hts_lens = [len(x) for x in inputs['hts']]
                input_logits = torch.split(outputs[-1].detach().cpu(), hts_lens, dim=0)
                add_logits_to_features(features, input_indices, input_logits)

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    loss1, loss2 = outputs[1], outputs[2]
                    loss1.detach().cpu().numpy()
                    loss2.detach().cpu().numpy()
                    writer.add_scalar("re_loss", loss1, global_step=num_steps)
                    writer.add_scalar("kd_loss", loss2, global_step=num_steps)
                    num_steps += 1

                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    # entropy = -(F.softmax(outputs[-1].detach().cpu(), dim=-1) * F.log_softmax(outputs[-1].detach().cpu(), dim=-1)).sum(1).mean()
                    re_loss, kd_loss, current_temperature, current_tradeoff = outputs[1].item(), outputs[2].item(), outputs[3], outputs[4]
                    print('===Epoch: {:d}, Step: {:d}, LM lr: {:.5e}, Classifier lr: {:.5e}, Total Loss: {:.5e}, RE Loss: {:.5e}, KD Loss: {:.5e}, Temperature: {:.5e}, Tradeoff: {:.5e}==='.format(
                        epoch, num_steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], loss.item(), re_loss, kd_loss, current_temperature, current_tradeoff), flush=True)

                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_output = evaluate(args, model, test_features, tag="test")

                    print('===performance===', flush=True)
                    print(dev_output, flush=True)
                    print(test_output, flush=True)

                    if dev_score > best_score:
                        best_score = dev_score
                        best_output = dev_output
                        best_test_score = test_score
                        nice_makedir('../../output/results')
                        _path = ('_'.join([str(_) for _ in
                                           [args.num_train_epochs,
                                            args.seed, args.learning_rate,
                                            args.lower_temperature,
                                            args.upper_temperature,
                                            args.loss_tradeoff]]))
                        _path = os.path.join('../../output/results', _path + '.json')
                        dump_json({'dev_f1': dev_score, 'test_f1': test_score}, _path)
                        # if args.save_test_pred != "":
                        #     pred = report(args, model, test_features)
                        #     with open(args.save_test_pred, "w") as fh:
                        #         json.dump(pred, fh)
                        # if args.save_ckpt != "":
                        #     torch.save(model.state_dict(), args.save_ckpt)
        print('----------\nBest Performance:', flush=True)
        print(best_output, flush=True)
        print(best_test_score, flush=True)
        print('----------\n', flush=True)
        return num_steps

    _path = ('_'.join([str(_) for _ in
                      [args.num_train_epochs,
                       args.seed, args.learning_rate,
                       args.lower_temperature,
                       args.upper_temperature,
                       args.loss_tradeoff]]))
    _path = os.path.join('../../output', _path)
    nice_makedir(_path)
    writer = SummaryWriter(_path)


    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, golds = [], []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[1].to(args.device),
                  'attention_mask': batch[2].to(args.device),
                  'entity_pos': batch[4],
                  'hts': batch[5],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[3]], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        "{}_f1".format(tag): f1 * 100,
    }
    return f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--root_path", default='./', type=str)
    parser.add_argument("--debug", default=1, type=int)
    parser.add_argument("--data_dir", default="./dataset/cdr", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="../../../model/gaojie/scibert", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--valid_file", default="valid.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_ckpt", default="./output/model.ckpt", type=str)
    parser.add_argument("--save_test_pred", default="./output/result.json", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_lr", default=1e-4, type=float,
                        help="The initial learning rate for Classifier.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--lower_temperature", default=1.0, type=float,
                        help="Lower bound of softmax temperature.")
    parser.add_argument("--upper_temperature", default=1.0, type=float,
                        help="Upper bound of softmax temperature.")
    parser.add_argument("--loss_tradeoff", default=1.0, type=float,
                        help="Tradeoff between RE and KD losses.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()

    print('args: ', flush=True)
    for arg in vars(args):
        print('    {}: {}'.format(arg, getattr(args, arg)), flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_gda

    train_file = os.path.join(args.root_path, args.data_dir, args.train_file)
    dev_file = os.path.join(args.root_path, args.data_dir, args.dev_file)
    test_file = os.path.join(args.root_path, args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, args.debug, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, args.debug, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, args.debug, max_seq_length=args.max_seq_length)

    ent_sum = 0
    men_sum = 0
    doc_sum = 0
    label_sum = 0
    for feature in train_features + dev_features + test_features:
        ent_sum += len(feature[0]['entity_pos'])
        men_sum += sum([len(_) for _ in feature[0]['entity_pos']])
        for label in feature[0]['labels']:
            if label == [0, 1]:
                label_sum += 1
        doc_sum += 1
    print(ent_sum / doc_sum)
    print(men_sum / ent_sum)
    print(label_sum / doc_sum)
    print(doc_sum)
    fffffffffffffffff

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels, lower_temperature=args.lower_temperature, upper_temperature=args.upper_temperature, loss_tradeoff=args.loss_tradeoff)
    model.to(0)

    if args.mode == "train":  # Training
        train(args, model, train_features, dev_features, test_features)
    if args.mode == 'test':  # Testing
        model.load_state_dict(torch.load(args.save_ckpt))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print('dev f1 and test f1 of model')
        print(dev_output)
        print(test_output)


if __name__ == "__main__":
    main()
