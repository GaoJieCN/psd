import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1, lower_temperature=1.0, upper_temperature=1.0, rkd_loss_tradeoff=1.0, ikd_loss_tradeoff=1.0):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.ikd_loss_fnt = nn.KLDivLoss(reduction='batchmean')
        self.rkd_loss_fnt = nn.SmoothL1Loss(beta=1.0)
        self.lower_temperature = lower_temperature
        self.upper_temperature = upper_temperature
        self.ikd_loss_tradeoff = ikd_loss_tradeoff
        self.rkd_loss_tradeoff = rkd_loss_tradeoff

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss, hts_lens = [], [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            hts_lens.append(hs.size()[0])
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, hts_lens

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                teacher_logits=None,
                teacher_divergences=None,
                current_step=None,
                total_steps=None,
                current_epoch=None,
                total_epochs=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts, hts_lens = self.get_hrt(sequence_output, attention, entity_pos, hts)
        device = sequence_output.device

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        divergences = []
        divergences_lens = []

        if labels is not None:
            current_temperature = self.upper_temperature - (self.upper_temperature - self.lower_temperature) * current_epoch / (total_epochs - 1.0)
            current_ikd_loss_tradeoff = self.ikd_loss_tradeoff * current_epoch / (total_epochs - 1.0)
            current_rkd_loss_tradeoff = self.rkd_loss_tradeoff * current_epoch / (total_epochs - 1.0)

            for one_logits, _label in zip(torch.split(logits, hts_lens, dim=0), labels):
                # one_logits.shape: (x, 97)
                # labels[i].shape: (x, 97), 与one_logits对应
                na_label = torch.tensor(_label)[:, 0].to(device)
                pos_label = 1 - na_label
                na_label_non_zero = torch.nonzero(na_label).reshape(-1)
                pos_label_none_zero = torch.nonzero(pos_label).reshape(-1)

                na_logits = one_logits[na_label_non_zero, :]
                pos_logits = one_logits[pos_label_none_zero, :]
                assert na_logits.shape[0] + pos_logits.shape[0] == one_logits.shape[0]

                def _kl_div(_logits_a, _logits_b, _temperature):
                    return F.kl_div(
                    F.log_softmax(_logits_a.unsqueeze(1).expand(-1, len(_logits_b), -1) / current_temperature,dim=-1),
                    F.softmax(_logits_b.unsqueeze(0).expand(len(_logits_a), -1, -1) / current_temperature, dim=-1),
                    reduction='none').sum(dim=-1)

                kl_div = (_kl_div(na_logits, pos_logits, current_temperature) +
                          _kl_div(pos_logits, na_logits, current_temperature).transpose(0, 1)) / 2

                kl_div = (kl_div / (kl_div.mean() + 1e-10)).view(-1, 1)
                divergences.append(kl_div)
                divergences_lens.append(kl_div.shape[0])
            divergences = torch.cat(divergences, dim=0)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits, divergences, divergences_lens)

        if labels is not None:
            ikd_loss = torch.tensor(0.0)
            rkd_loss = torch.tensor(0.0)
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            re_loss = self.loss_fnt(logits.float(), labels.float())
            if teacher_logits is not None:
                teacher_logits = torch.cat(teacher_logits, dim=0).to(logits)
                ikd_loss = self.ikd_loss_fnt(F.log_softmax(logits/current_temperature, dim=1),
                                             F.softmax(teacher_logits/current_temperature, dim=1))
            if teacher_divergences is not None:
                teacher_divergences = torch.cat(teacher_divergences, dim=0).to(divergences)
                rkd_loss = self.rkd_loss_fnt(divergences, teacher_divergences)
            loss = re_loss + current_ikd_loss_tradeoff * ikd_loss + current_rkd_loss_tradeoff * rkd_loss
            output = (loss.to(sequence_output), re_loss.to(sequence_output),
                      ikd_loss.to(sequence_output), rkd_loss.to(sequence_output),
                      current_ikd_loss_tradeoff, current_rkd_loss_tradeoff, current_temperature) + output

        return output
