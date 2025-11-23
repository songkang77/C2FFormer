from data import get_datasets_path,get_datasets_path_MAR,get_datasets_path_onepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import math
from functools import partial
from basic_var import AdaLNSelfAttn,AdaLNBeforeHead
from datetime import datetime

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class STSGM(nn.Module):
    
    def __init__(
        self, args, 
        depth=128, embed_dim=7, num_heads=7, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,num_classes=1,        
        ):
        super().__init__()
        self.args = args
        
        if args.name =='electricity':
            if args.window_size == 96:
                self.pool_sizes = [24, 32, 48, 96]
            elif args.window_size == 48:
                self.pool_sizes = [12, 16, 24, 48]                                                                              
            elif args.window_size == 24:
                self.pool_sizes = [6, 8, 12, 24]                                                                    
        if args.name == 'beijing':
            if args.window_size == 96:
                self.pool_sizes = [24, 32, 48, 96]
            elif args.window_size == 48:
                self.pool_sizes = [12, 16, 24, 48]
            elif args.window_size == 24:
                self.pool_sizes = [6, 8, 12, 24]
        if args.name == 'italy':
            if args.window_size == 96:
                self.pool_sizes = [24, 32, 48, 96]
            elif args.window_size == 48:
                self.pool_sizes = [12, 16, 24, 48] 
            elif args.window_size == 24:
                self.pool_sizes = [6, 8, 12, 24]
        if args.name == 'pedestrian':
            if args.window_size == 24:
                self.pool_sizes = [6, 8, 12, 24]
        if args.name == 'physionet_2012':
            if args.window_size == 48:
                self.pool_sizes = [12, 16, 24, 48] 
        if args.name == 'pems':
            if args.window_size == 24:
                self.pool_sizes = [6, 8, 12, 24]

        self.L = sum(pn for pn in self.pool_sizes)  # Total number of pooled features
        
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        self.C = embed_dim
        self.D = embed_dim
        self.B = args.batch_size
        self.num_heads = num_heads
        self.first_l = self.pool_sizes[0]
        bg,ed = (0,self.L)
        self.ed = ed
        self.cond_drop_rate = cond_drop_rate
        
        self.train_loss = nn.MSELoss(reduction='mean')
        self.norm_eps = norm_eps
        self.loss_weight = torch.ones(1, self.L, device='cuda') / self.L
        
    
        #attention mask  
        d: torch.Tensor = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.pool_sizes)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)#block-wise mask
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        attn_mask = self.attn_bias_for_masking[:,:,:ed,:ed]
        self.attn_mask = attn_mask
        self.attn_mask = self.attn_mask.to('cuda:0')
        print(f'atten_mask shape :',attn_mask.shape)
        
        #class embedding  
        init_std = math.sqrt(1 / self.C / 6)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device='cuda')
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C)) #(1,1,1024)
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
       
        #absolute position embedding  
        pos_1LC = []
        for i, pn in enumerate(self.pool_sizes):
            pe = torch.empty(1, pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.pool_sizes), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=False, fused_if_available=False,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={False} ({sum(b.attn.using_flash for b in self.blocks)}/{depth}), fused_if_available={False} (fusing_add_ln={sum(fused_add_norm_fns)}/{depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C,self.C)
        
        B = args.batch_size 
        label_B = torch.ones(B, device='cuda', dtype=torch.long) 
        label_B = torch.where(torch.rand(B, device='cuda') < self.cond_drop_rate, self.num_classes, label_B)
        self.label_B = label_B
        
    def dataloader(self, path):
        if self.args.MAR:
            # BWM
            train_set,val_set_X,val_set_X_ori,test_X,test_X_ori,test_indicating_mask = get_datasets_path_MAR(path,self.args.name,self.args.rate)
        else:
            # RM
            train_set,val_set_X,val_set_X_ori,test_X,test_X_ori,test_indicating_mask=get_datasets_path(path,self.args.name)# *
        if self.args.onepiece:
            test_X_05,test_X_ori_05,test_indicating_mask_05,test_X_01,test_X_ori_01,test_indicating_mask_01 = get_datasets_path_onepiece(path,self.args.name)# *


        if val_set_X.shape[0] > self.B:
           
            indices = torch.randperm(val_set_X.shape[0])[:self.B]
            
            val_set_X = val_set_X[indices]
            val_set_X_ori = val_set_X_ori[indices]
        elif val_set_X.shape[0] < self.B:
           
            num_repeats = (self.B + val_set_X.shape[0] - 1) // val_set_X.shape[0]  
            
            repeated_val_set_X = val_set_X.repeat(num_repeats, 1, 1)
            repeated_val_set_X_ori = val_set_X_ori.repeat(num_repeats, 1, 1)
            
            val_set_X = repeated_val_set_X[:self.B]     
            val_set_X_ori = repeated_val_set_X_ori[:self.B]  
        
        
        if test_X.shape[0] > self.B:
           
            indices = torch.randperm(test_X.shape[0])[:self.B]
            
            test_X = test_X[indices]
            test_X_ori = test_X_ori[indices]
            test_indicating_mask = test_indicating_mask[indices]
        elif test_X.shape[0] < self.B:
            
            num_repeats = (self.B + test_X.shape[0] - 1) // test_X.shape[0]  
            
            repeated_test_X = test_X.repeat(num_repeats, 1, 1)
            repeated_test_X_ori = test_X_ori.repeat(num_repeats, 1, 1)
            repeated_test_indicating_mask = test_indicating_mask.repeat(num_repeats, 1, 1)
           
            test_X = repeated_test_X[:self.B]
            test_X_ori = repeated_test_X_ori[:self.B]     
            test_indicating_mask = repeated_test_indicating_mask[:self.B]  

        if self.args.onepiece:
            
            if test_X_05.shape[0] > self.B:
                print(test_X_05.shape[0])
            
                indices = torch.randperm(test_X_05.shape[0])[:self.B]
            
                test_X_05 = test_X_05[indices]
                test_X_ori_05 = test_X_ori_05[indices]
                test_indicating_mask_05 = test_indicating_mask_05[indices]
                test_X_01 = test_X_01[indices]
                test_X_ori_01 = test_X_ori_01[indices]
                test_indicating_mask_01 = test_indicating_mask_01[indices]

            self.test_X_05 = torch.tensor(test_X_05).to('cuda:0')
            self.test_X_ori_05 = torch.tensor(test_X_ori_05).to('cuda:0')
            self.test_indicating_mask_05 = torch.tensor(test_indicating_mask_05).to('cuda:0')

            self.test_X_01 = torch.tensor(test_X_01).to('cuda:0')
            self.test_X_ori_01 = torch.tensor(test_X_ori_01).to('cuda:0')
            self.test_indicating_mask_01 = torch.tensor(test_indicating_mask_01).to('cuda:0')
        
        self.train_set = torch.tensor(train_set).to('cuda:0')
        self.val_set_X = torch.tensor(val_set_X).to('cuda:0')
        self.val_set_X_ori = torch.tensor(val_set_X_ori).to('cuda:0')
        self.test_X = torch.tensor(test_X).to('cuda:0')
        self.test_X_ori =torch.tensor(test_X_ori).to('cuda:0')
        self.test_indicating_mask = torch.tensor(test_indicating_mask).to('cuda:0')
        
    def inference(self, x: torch.Tensor):
        if self.args.batch_size == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        x = x.to('cuda:0')
        ori_x = x.clone()
       
        if torch.isnan(x).any():
           
            for b in range(B):
                for c in range(x.shape[2]):
                  
                    seq = x[b, :, c]
                    mask = torch.isnan(seq)
                   
                    if mask.any():
                       
                        valid_indices = torch.nonzero(~mask).squeeze()
                        valid_values = seq[~valid_indices]
                        
                        
                        nan_indices = torch.nonzero(mask).squeeze()
                        
                       
                        if valid_indices.numel() == 0:
                            seq[nan_indices] = 0.0
                            continue
                            
                        
                        if nan_indices.numel() == 1:
                            idx = nan_indices.item()
                            left_idx = valid_indices[valid_indices < idx]
                            right_idx = valid_indices[valid_indices > idx]

                            if left_idx.numel() > 0 and right_idx.numel() > 0:
                                   
                                left_idx = left_idx[-1]
                                right_idx = right_idx[0]
                                left_val = seq[left_idx]
                                right_val = seq[right_idx]
                                  
                                weight = (idx - left_idx).float() / (right_idx - left_idx).float()
                                seq[idx] = left_val + weight * (right_val - left_val)
                            elif left_idx.numel() > 0:
                                  
                                seq[idx] = seq[left_idx[-1]]
                            elif right_idx.numel() > 0:
                                   
                                seq[idx] = seq[right_idx[0]]
                            else:
                                   
                                seq[idx] = 0.0
                        else:
                            for idx in nan_indices:
                               
                                left_idx = valid_indices[valid_indices < idx]
                                right_idx = valid_indices[valid_indices > idx]

                                if left_idx.numel() > 0 and right_idx.numel() > 0:
                                   
                                    left_idx = left_idx[-1]
                                    right_idx = right_idx[0]
                                    left_val = seq[left_idx]
                                    right_val = seq[right_idx]
                                  
                                    weight = (idx - left_idx).float() / (right_idx - left_idx).float()
                                    seq[idx] = left_val + weight * (right_val - left_val)
                                elif left_idx.numel() > 0:
                                   
                                    seq[idx] = seq[left_idx[-1]]
                                elif right_idx.numel() > 0:
                                   
                                    seq[idx] = seq[right_idx[0]]
                                else:
                        
                                    seq[idx] = 0.0
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=(self.pool_sizes[0],),mode='area')
        x = x.permute(0, 2, 1)
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        
        cond_BD = self.class_emb(self.label_B)
       
        next_token_map = x + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.pool_sizes):
           
            
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            
            if si != len(self.pool_sizes) - 1:
                next_token_map = self.get_next_input(x,ori_x,si)
        
        for b in self.blocks: b.attn.kv_caching(False)
        x.to('cuda:0')
        return x
    
    def get_next_input(self, x: torch.Tensor,ori_x: torch.Tensor,si):

        # x: (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        # print("get_next_input x shape before interp:", x.shape )
        ori_x = ori_x.permute(0, 2, 1)
        # print("get_next_input ori_x shape before interp:", ori_x.shape )
        # f = F.interpolate(x, size=(self.pool_sizes[si+1],),mode='linear')
        f = F.interpolate(x, size=(self.pool_sizes[-1],),mode='linear')
        # print("get_next_input f shape after interp:", f.shape )
        nan_mask = torch.isnan(ori_x)
        f = torch.where(nan_mask, f, ori_x)
        f = F.interpolate(f, size=(self.pool_sizes[si+1],),mode='area')
        # print("get_next_input f shape after second interp:", f.shape )
        return f.permute(0, 2, 1)
        
    def mask_random_sample(self, x: torch.Tensor, pool_size: int) -> torch.Tensor:

        B, C, L = x.shape

        # 
        if pool_size > L:
            raise ValueError(f"pool_size ({pool_size}) cannot be larger than sequence length ({L})")

        # 
        result = torch.zeros((B, C, pool_size), device=x.device)

        # 
        keep_ratio = pool_size / L # 24/96=4

        # 
        for b in range(B):
            for c in range(C):
                
               
                mask = torch.rand(L, device=x.device) < keep_ratio 

   
                if mask.sum() > pool_size:
                    # 
                    selected_indices = torch.where(mask)[0]
                    remove_count = mask.sum() - pool_size
                    remove_indices = selected_indices[torch.randperm(len(selected_indices))[:remove_count]]# 
                    mask[remove_indices] = False
                elif mask.sum() < pool_size:
                    # 
                    unselected_indices = torch.where(~mask)[0]
                    add_count = pool_size - mask.sum()
                    add_indices = unselected_indices[torch.randperm(len(unselected_indices))[:add_count]]
                    mask[add_indices] = True

                # 
                selected_indices = torch.where(mask)[0]
                # 
                selected_indices = torch.sort(selected_indices)[0]

                # 
                result[b, c] = x[b, c][selected_indices]
        print(result.shape)# 2,13,24
        return result
    
    def dataprocess_train(self, x):

        if x.shape[0] > self.B:
            # 
            indices = torch.randperm(x.shape[0])[:self.B]
            # 
            x = x[indices]
        elif x.shape[0] < self.B:
            # 
            num_repeats = (self.B + x.shape[0] - 1) // x.shape[0]  # 
            # 
            repeated_x = x.repeat(num_repeats, 1, 1)
            # 
            x = repeated_x[:self.B]
        

        ori_x = x.clone()  # Save the original input for later use
        x = x.permute(0, 2, 1)  # (B, C, L)
        pooled_features = []

        for pool_size in self.pool_sizes:
            if self.args.MAR :
                sampled = F.interpolate(x, size=pool_size, mode='area')
            else:
                sampled = self.mask_random_sample(x, pool_size)
            pooled_features.append(sampled)
            
        x = torch.cat(pooled_features, dim=2)
        x = x.permute(0, 2, 1)
        return ori_x,x
    
    def dataprocess_val(self,x):
        ori_x = x.clone()
        ori_x = ori_x.to('cuda:0')# Save the original input for later use
        x = x.permute(0, 2, 1)  # (B, C, L)
        pooled_features = []
        sampled = F.interpolate(x, size=24, mode='linear')
        pooled_features.append(sampled)
        x = torch.cat(pooled_features, dim=2)  # (B, C, L)
        
        # (B, L, C)
        x = x.permute(0, 2, 1)
        return ori_x,x
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
    
    def get_logits(self, x: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        return self.head(self.head_nm(x.float(), cond_BD).float()).float()
    
    def forward(self, ori_x, x: torch.Tensor):
        B = x.shape[0]
        x = x.to('cuda:0')
        ori_x = ori_x.to('cuda:0')
        xloss  = x.clone()
        
        cond_BD = self.class_emb(self.label_B)
      
        x += self.lvl_embed(self.lvl_1L[:, :self.ed].expand(B, -1)) + self.pos_1LC[:, :self.ed] # lvl: BLC;  pos: 1LC
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        # print("x shape before blocks:", x.shape)# [2, 200, 13] 200=24+32+48+96  200个token

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=self.attn_mask)
            

        
        
        return x,xloss
    
    

