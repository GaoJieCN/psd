import argparse
import os
import time
from tqdm import tqdm
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
from prepro import read_docred
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
    def finetune(features, optimizer, total_epochs):
        best_score = -1
        best_output = {}
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
        train_iterator = range(int(total_epochs))
        total_steps = int(len(train_dataloader) * total_epochs // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))

        nice_makedir('../../output/logs')
        _path = ('_'.join([str(_) for _ in
                           [args.num_train_epochs,
                            args.learning_rate,
                            args.lower_temperature,
                            args.upper_temperature,
                            args.ikd_loss_tradeoff,
                            args.rkd_loss_tradeoff]]))
        _path1 = os.path.join('../../output/logs', _path + 'loss.log')
        loss_writer = SummaryWriter(_path1)
        _path1 = os.path.join('../../output/logs', _path + 're_loss.log')
        re_loss_writer = SummaryWriter(_path1)
        _path1 = os.path.join('../../output/logs', _path + 'ikd_loss.log')
        ikd_loss_writer = SummaryWriter(_path1)
        _path1 = os.path.join('../../output/logs', _path + 'rkd_loss.log')
        rkd_loss_writer = SummaryWriter(_path1)

        current_step = 0
        for current_epoch in tqdm(train_iterator):
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[1].to(args.device),
                          'attention_mask': batch[2].to(args.device),
                          'labels': batch[3],
                          'entity_pos': batch[4],
                          'hts': batch[5],
                          'teacher_logits': batch[6],
                          'teacher_divergences': batch[7],
                          'current_step': current_step,
                          'total_steps': total_steps,
                          'current_epoch': current_epoch,
                          'total_epochs': total_epochs,
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()

                input_indices = batch[0]
                hts_lens = [len(x) for x in inputs['hts']]
                divergences_lens = outputs[-1]
                input_logits = torch.split(outputs[-3].detach().cpu(), hts_lens, dim=0)
                input_divergences = torch.split(outputs[-2].detach().cpu(), divergences_lens, dim=0)
                add_logits_to_features(features, input_indices, input_logits, input_divergences)

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    current_step += 1

                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and current_step % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    re_loss, ikd_loss, rkd_loss = outputs[1].item(), outputs[2].item(), outputs[3].item()
                    ikd_loss_tradeoff, rkd_loss_tradeoff, temperature = outputs[4], outputs[5], outputs[6]
                    loss_writer.add_scalar("loss", loss.item(), global_step=current_step)
                    re_loss_writer.add_scalar("loss", re_loss, global_step=current_step)
                    ikd_loss_writer.add_scalar("loss", ikd_loss_tradeoff * ikd_loss, global_step=current_step)
                    rkd_loss_writer.add_scalar("loss", rkd_loss_tradeoff * rkd_loss, global_step=current_step)

                    print('\n---------------------------------------------------------', flush=True)
                    print('Epoch: {:d}, Step: {:d}, LM lr: {:.5e}, Classifier lr: {:.5e}'.format(current_epoch, current_step, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']), flush=True)
                    print('Total Loss: {:.5e}, RE Loss: {:.5e}, IKD Loss: {:.5e}, RKD Loss: {:.5e}'.format(loss.item(), re_loss, ikd_loss, rkd_loss), flush=True)
                    print('IKD Loss Tradeoff: {:.5e}, RKD Loss Tradeoff: {:.5e}, Temperature'.format(ikd_loss_tradeoff, rkd_loss_tradeoff, temperature), flush=True)
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    print('===performance===', flush=True)
                    print(dev_output, flush=True)
                    if dev_score > best_score:
                        best_score = dev_score
                        best_output = dev_output
                        nice_makedir('../../output/results')
                        _path = ('_'.join([str(_) for _ in
                                           [args.num_train_epochs,
                                            args.seed, args.learning_rate,
                                            args.lower_temperature,
                                            args.upper_temperature,
                                            args.ikd_loss_tradeoff,
                                            args.rkd_loss_tradeoff]]))
                        _path = os.path.join('../../output/results', _path + '.json')
                        dump_json(best_output, _path)
                        pred = report(args, model, test_features)
                        with open('../../output/result_{}.json'.format(args.seed), "w") as fh:
                            json.dump(pred, fh)

        print('----------\nBest Performance:', flush=True)
        print(best_output, flush=True)
        print('----------\n', flush=True)

    _path = ('_'.join([str(_) for _ in
                      [args.num_train_epochs,
                       args.seed, args.learning_rate,
                       args.lower_temperature,
                       args.upper_temperature,
                       args.ikd_loss_tradeoff,
                       args.rkd_loss_tradeoff]]))
    _path = os.path.join('../../output', _path)
    nice_makedir(_path)

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
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

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if tag == 'train':
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, os.path.join(args.root_path, args.data_dir), args.train_file)
    elif tag == 'dev':
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, os.path.join(args.root_path, args.data_dir), args.dev_file)

    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
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

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--root_path", default='./', type=str)
    parser.add_argument("--debug", default=1, type=int)
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="../../model/gaojie/local-bert-base-cased", type=str)

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
    parser.add_argument("--ikd_loss_tradeoff", default=1.0, type=float,
                        help="Tradeoff between RE and KD losses.")
    parser.add_argument("--rkd_loss_tradeoff", default=1.0, type=float,
                        help="Tradeoff between RE and KD losses.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()

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

    read = read_docred

    train_file = os.path.join(args.root_path, args.data_dir, args.train_file)
    dev_file = os.path.join(args.root_path, args.data_dir, args.dev_file)
    test_file = os.path.join(args.root_path, args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, args.debug, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, args.debug, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, args.debug, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels,
                       lower_temperature=args.lower_temperature,
                       upper_temperature=args.upper_temperature,
                       ikd_loss_tradeoff=args.ikd_loss_tradeoff,
                       rkd_loss_tradeoff=args.rkd_loss_tradeoff)
    model.to(0)

    if args.mode == "train":  # Training
        train(args, model, train_features, dev_features, test_features)
    if args.mode == 'test':  # Testing
        model.load_state_dict(torch.load(args.save_ckpt))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred, logits = report(args, model, test_features)
        with open(args.save_test_pred, "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
