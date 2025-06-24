import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoModel
from TorchCRF import CRF

class BEA(nn.Module):
    def __init__(self, args, output_dim):
        super(BEA, self).__init__()
        self.args = args
        self.score_pred = score_pred(self.args, output_dim=output_dim)

    def forward(self, conversation, response, mask=None):
        pred_score = self.score_pred(conversation, response, mask)
        return pred_score


class score_pred(nn.Module):
    def __init__(self, args, output_dim):
        super(score_pred, self).__init__()
        self.args = args

        self.essay_prompt_process_list = nn.ModuleList([
            essay_prompt_process(self.args) for _ in range(output_dim)
        ])

        self.final_pred_list = nn.ModuleList([
            trait_process_classification(self.args) for _ in range(output_dim)
        ])

        self.crf_pred = MultiTaskWithCRF()

        self.bilstm = nn.LSTM(
            input_size=3,  # 与 cross_att_out.shape[-1] 保持一致
            hidden_size=10,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.bilstm_fc = nn.Linear(in_features=10 * 2, out_features=3)

    def forward(self, conversation, response, labels, mask=None):
        cross_att_out = []
        for process in self.essay_prompt_process_list:
            process_out = process(conversation, response, mask)
            cross_att_out.append(process_out)
        cross_att_out = torch.stack(cross_att_out, dim=1)

        emissions = []
        for index, pred in enumerate(self.final_pred_list):
            pred_score = pred(index, cross_att_out)
            emissions.append(pred_score)

        emissions = torch.stack(emissions, dim=1) # [B, 4, 3]

        lstm_out, _ = self.bilstm(emissions) # [B, 4, 3]
        preds = self.bilstm_fc(lstm_out)

        crf = self.crf_pred(emissions, labels) # [B, 4]
        # return emissions, 0
        return emissions, crf


class trait_process_classification(nn.Module):
    def __init__(self, args):
        super(trait_process_classification, self).__init__()
        self.args = args
        self.att = Attention(input_dim=args.embed_dim, output_dim=args.embed_dim, isMultiHead=False)
        self.final_dense = nn.Sequential(
            nn.Linear(in_features=args.embed_dim * 2, out_features=args.embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=args.embed_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=3)  # 输出层
        )

    def forward(self, index, cross_att_out):
        non_target_rep = torch.cat((cross_att_out[:, :index, :], cross_att_out[:, index+1:, :]), dim=-2)
        target_rep = cross_att_out[:, index:index+1]
        att_out = self.att(target_rep, cross_att_out, cross_att_out) # [batch_size, 1, embed_dim]
        att_cat = torch.cat((target_rep, target_rep), dim=-1).squeeze(-2) # [batch_size, embed_dim*2]
        pred_score = self.final_dense(att_cat) # [batch_size, 1, 3]
        return pred_score


class essay_prompt_process(nn.Module):
    def __init__(self, args):
        super(essay_prompt_process, self).__init__()
        self.args = args

        self.mul_att = Attention(input_dim=args.embed_dim, output_dim=args.embed_dim, isMultiHead=True)
        self.dense = nn.Sequential(
            nn.Linear(in_features=args.embed_dim, out_features=args.embed_dim),
            nn.ReLU(),
        )

    def forward(self, conversation, response, mask=None):
        # conversation / response: [batch_size, para_num, embed_dim]
        mh_at = self.mul_att(response, conversation, conversation) # [batch_size, para_num, embed_dim] same as prompt_feat
        mh_at = mh_at.squeeze(1)
        mh_at = self.dense(mh_at) + mh_at
        return mh_at


class MultiTaskWithCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.crf = CRF(num_labels=3)

    def forward(self, emissions, labels=None):
        mask_crf = torch.ones(emissions.shape[:2], dtype=torch.bool).to(emissions.device)
        if labels is not None:
            # labels: [B, 4], 每个位置是 0, 1, 2 的标签
            log_likelihood = self.crf(emissions, labels, mask=mask_crf)
            return -log_likelihood  # loss
        else:
            pred = self.crf.viterbi_decode(emissions, mask=mask_crf)     # List[List[int]] of size B
            return torch.tensor(pred)



class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)  # 投影层
        self.v = nn.Parameter(torch.randn(hidden_dim))  # 可学习的注意力向量

    def forward(self, x, mask=None):
        # x: [dim_1, dim_2, dim_3]
        x_proj = torch.tanh(self.proj(x))  # activative func [dim_1, dim_2, dim_3]
        scores = torch.matmul(x_proj, self.v)  # att score [dim_1, dim_2]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('1e-9'))
        weights = F.softmax(scores, dim=1)    # softmax [dim_1, dim_2]

        # [dim_1, dim_3]
        outputs = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return outputs


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, isMultiHead, num_heads=8):
        super(Attention, self).__init__()
        self.isMultiHead = isMultiHead
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        if isMultiHead:
            assert output_dim % num_heads == 0

        self.projection_dim = output_dim // num_heads
        self.wq = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.wk = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.wv = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.dense = nn.Linear(in_features=output_dim, out_features=output_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.wq, self.wk, self.wv, self.dense]:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1)) # q*k
        logits = matmul_qk / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32)) # q*k / sqrt(d)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-1e9')) # mask on
        attention_weights = nn.functional.softmax(logits, dim=-1) # softmax
        output = torch.matmul(attention_weights, value) # ()*v
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        if self.isMultiHead:
            query = self.split_heads(query, batch_size)
            key = self.split_heads(key, batch_size)
            value = self.split_heads(value, batch_size)

        attention, _ = self.scaled_dot_product_attention(query, key, value, mask)
        if self.isMultiHead:
            attention = attention.permute(0, 2, 1, 3)
            attention = torch.reshape(attention, (batch_size, -1, self.output_dim))
        output = self.dense(attention)
        return output


if __name__ == '__main__':
    print(1)
