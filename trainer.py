import torch
from lr_control import filter_params
from amp_sc import AmpOptimizer
from model import STSGM
from functools import partial
from plot import visualize_tensors
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
# this is the trainer class
class GMTrainer(object):
    def __init__(self, args):
        self.args = args
        self.model = STSGM(args,shared_aln=True).cuda()
        self.model.init_weights(init_std=-1)
        self.model.dataloader(args.path)
        var_opt = self.var_optim = self.build_optimizer()
    
    def build_optimizer(self):
        names, paras, para_groups = filter_params(self.model, nowd_keys={
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
        })
        opt_clz = {
            'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=self.args.afuse),
            'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=self.args.afuse),
        }[self.args.opt.lower().strip()]
        opt_kw = dict(lr=self.args.tlr, weight_decay=0)
        print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')

        var_optim = AmpOptimizer(
            mixed_precision=self.args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
            grad_clip=self.args.tclip, n_gradient_accumulation=self.args.ac
        )
        return var_optim
    
    def train(self):
        
        x = self.model.train_set
        # print("train x shape:", x.shape) # train x shape: torch.Size([58, 96, 13])
        ori_x, x = self.model.dataprocess_train(x)
        x ,xloss = self.model.forward(ori_x, x)
        loss = self.model.train_loss(x, xloss)
        lw = self.model.loss_weight
        loss = loss.mul(lw).sum(dim=-1).mean()
        grad_norm, scale_log2 =  self.var_optim.backward_clip_step(loss=loss,stepping=True)
        grad_norm = grad_norm.item()
            
            
        return loss, grad_norm, scale_log2
    
    def val(self):
        x = self.model.val_set_X
        x = self.model.inference(x)
        if len(self.model.val_set_X_ori.shape) == 2:
            self.model.val_set_X_ori.unsqueeze_(0)
        loss = calc_mse(x,self.model.val_set_X_ori)
        
        return loss
    
    def test(self):
        x = self.model.test_X
        x= self.model.inference(x)
        if len(self.model.test_X_ori.shape) == 2:
            self.model.test_X_ori.unsqueeze_(0)
        if len(self.model.test_indicating_mask.shape) == 2:
            self.model.test_indicating_mask.unsqueeze_(0)
        loss_mse = calc_mse(x,self.model.test_X_ori,self.model.test_indicating_mask)
        loss_mae = calc_mae(x,self.model.test_X_ori,self.model.test_indicating_mask)
        loss_mre = calc_mre(x,self.model.test_X_ori,self.model.test_indicating_mask)
        # visualize_tensors(x=x, ori_x=self.model.test_X_ori,masks=self.model.test_indicating_mask,path=self.args.result_pic_path)
        return loss_mae,loss_mse,loss_mre
    def test_onepiece_05(self):
        x_05 = self.model.test_X_05
        x_05 = self.model.inference(x_05)
        if len(self.model.test_X_ori_05.shape) == 2:
            self.model.test_X_ori_05.unsqueeze_(0)
        if len(self.model.test_indicating_mask_05.shape) == 2:
            self.model.test_indicating_mask_05.unsqueeze_(0)
        loss_mse = calc_mse(x_05,self.model.test_X_ori_05,self.model.test_indicating_mask_05)
        loss_mae = calc_mae(x_05,self.model.test_X_ori_05,self.model.test_indicating_mask_05)
        loss_mre = calc_mre(x_05,self.model.test_X_ori_05,self.model.test_indicating_mask_05)
        path1 = 'result_pic' + '_rate0.5' + '/' + self.args.name
        visualize_tensors(x=x_05, ori_x=self.model.test_X_ori_05,masks=self.model.test_indicating_mask_05,path=path1)
        return loss_mae,loss_mse,loss_mre

    def test_onepiece_01(self):
        x_01 = self.model.test_X_01
        x_01 = self.model.inference(x_01)
        if len(self.model.test_X_ori_01.shape) == 2:
            self.model.test_X_ori_01.unsqueeze_(0)
        if len(self.model.test_indicating_mask_01.shape) == 2:
            self.model.test_indicating_mask_01.unsqueeze_(0)
        loss_mse = calc_mse(x_01,self.model.test_X_ori_01,self.model.test_indicating_mask_01)
        loss_mae = calc_mae(x_01,self.model.test_X_ori_01,self.model.test_indicating_mask_01)
        loss_mre = calc_mre(x_01,self.model.test_X_ori_01,self.model.test_indicating_mask_01)
        path1= 'result_pic_rate0.1/'+slef.args.name
        visualize_tensors(x=x_01, ori_x=self.model.test_X_ori_01,masks=self.model.test_indicating_mask_01,path=path1)
        return loss_mae,loss_mse,loss_mre
    def state_dict(self):
        return self.model.state_dict()
        
     
    def load_state_dict(self,state):
        self.model.load_state_dict(state_dict=state,strict=True)
        
        

    

