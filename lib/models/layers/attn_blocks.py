import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from ...utils.ce_utils import generate_mask
import torch.nn.functional as F
from lib.models.layers.attn import Attention


def candidate_elimination_based_divide_prediction(max_positions: torch.Tensor, foreground_probabilities, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor):
    lens_s = tokens.shape[1] - lens_t
    bs, N , D = tokens.shape
    grid_size = 16

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None, None

    #contextual zone
    masks_11x11 = torch.zeros([bs, grid_size, grid_size], device=tokens.device, dtype=torch.bool)

    for b in range(bs):
        center_x, center_y = max_positions[b]
        x_start_11, x_end_11 = max(center_x - 5, 0), min(center_x + 6, grid_size)
        y_start_11, y_end_11 = max(center_y - 5, 0), min(center_y + 6, grid_size)
        masks_11x11[b, x_start_11:x_end_11, y_start_11:y_end_11] = 1

    probabilities_outside = foreground_probabilities.squeeze(-1).clone()

    probabilities_outside = probabilities_outside.masked_fill(masks_11x11.flatten(1) == 1, float('inf'))


    sorted_prob, indices = torch.sort(probabilities_outside, dim=1, descending=True)
    topk_prob, topk_idx = sorted_prob[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_prob[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index, topk_idx
class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,currect_layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.division_layer = 4
        with_division = currect_layer == self.division_layer
        self.divide_predict = nn.Sequential(
            nn.Linear(dim * 3, 768),
            nn.GELU(),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, 1)
        ) if with_division else None

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, keep_ratio_search=None, attn_mask=None, tgt_type=None,learn_template_mask=None):

        removed_index_search = None
        if self.divide_predict:
            B, N, C = x.shape
            if tgt_type == 'allmax':
                tgt_rep = x[:, global_index_template.shape[1]//2:-global_index_search.shape[1]]
                tgt_rep = F.adaptive_max_pool1d(tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
            elif tgt_type == 'roimax':
                intial_tgt_rep = x[:, :global_index_template.shape[1]//2] * learn_template_mask[0].unsqueeze(-1)
                intial_tgt_rep = F.adaptive_max_pool1d(intial_tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)
                dynamic_tgt_rep = x[:,  global_index_template.shape[1] // 2:-global_index_search.shape[1]] * learn_template_mask[1].unsqueeze(-1)
                dynamic_tgt_rep = F.adaptive_max_pool1d(dynamic_tgt_rep.transpose(1, 2), output_size=1).transpose(1, 2)

            initial_tgt_rep = intial_tgt_rep.expand(-1, global_index_search.shape[1], -1)
            dynamic_tgt_rep = dynamic_tgt_rep.expand(-1, global_index_search.shape[1], -1)
            divide_prediction = self.divide_predict(torch.cat((x[:, -global_index_search.shape[1]:], initial_tgt_rep, dynamic_tgt_rep), dim=-1))

            grid_size = 16
            foreground_probabilities = torch.sigmoid(divide_prediction)
            foreground_grid = foreground_probabilities.view(B, 1, grid_size, grid_size)

            conv_filter = torch.ones(1, 1, 3, 3, device=foreground_grid.device)
            filtered_sum = torch.nn.functional.conv2d(foreground_grid, conv_filter, stride=1, padding=1)

            max_vals, max_indices = torch.max(filtered_sum.view(B, -1), dim=1)
            max_positions = torch.stack([max_indices // grid_size, max_indices % grid_size], dim=1)

            masks_7x7 = torch.zeros([B, grid_size, grid_size], device=x.device, dtype=torch.bool)
            foreground = (foreground_probabilities > 0.5).long()

            for b in range(B):
                center_x, center_y = max_positions[b]

                #7*7
                x_start_7, x_end_7 = max(center_x - 3, 0), min(center_x + 4, grid_size)
                y_start_7, y_end_7 = max(center_y - 3, 0), min(center_y + 4, grid_size)
                masks_7x7[b, x_start_7:x_end_7, y_start_7:y_end_7] = 1

            foreground_mask = masks_7x7.flatten(1)
            selected_foreground = foreground * foreground_mask.unsqueeze(-1)

            if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
                keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                x, global_index_search, removed_index_search, topk_idx = candidate_elimination_based_divide_prediction(max_positions, foreground_probabilities, x, global_index_template.shape[1],
                                                                                                                keep_ratio_search,
                                                                                                                global_index_search)
                if topk_idx is not None:
                    selected_foreground = selected_foreground.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, selected_foreground.shape[-1]))

            selected_foreground = selected_foreground.to(torch.int64)
            decision = torch.zeros(selected_foreground.size(0), selected_foreground.size(1), 2,
                                   device=selected_foreground.device)  # Shape: [2, 4, 2]
            decision.scatter_(2, selected_foreground, 1)

            blank_policy = torch.zeros(B, global_index_search.shape[1], 2, dtype=divide_prediction.dtype,
                                       device=divide_prediction.device)
            search_policy = torch.cat([blank_policy, decision], dim=-1)

            template_policy = torch.zeros(B, global_index_template.shape[1]//2, 4, dtype=divide_prediction.dtype,
                                          device=divide_prediction.device)
            template_policy[:, :, 0] = 1
            dynamic_policy = torch.zeros(B, global_index_template.shape[1]//2, 4, dtype=divide_prediction.dtype,
                                          device=divide_prediction.device)
            dynamic_policy[:, :, 1] = 1
            policy = []
            policy.append(template_policy)
            policy.append(dynamic_policy)
            policy.append(search_policy)
            attn_mask = generate_mask(policy)
            '''background_count = decision[:, :, 0].sum(dim=1)
            foreground_count = decision[:, :, 1].sum(dim=1)
            for i in range(decision.shape[0]):
                print(
                    f"Batch {i + 1} - Background tokens: {background_count[i]}, Foreground tokens: {foreground_count[i]}")'''
        x_attn, attn = self.attn(self.norm1(x), mask, True,attn_mask=attn_mask)
        x = x + self.drop_path(x_attn)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn, attn_mask


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
