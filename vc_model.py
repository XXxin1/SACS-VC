from network import Generator, Discriminator
from utils.common_util import *
import copy
from torch import nn
from utils.patchnce import PatchNCELoss



class VCModel(nn.Module):
    def __init__(self, config, inference_mask=False):
        super(VCModel, self).__init__()
        self.config = config
        self.inference_mask = inference_mask
        self.__build_model()

    def __build_model(self):
        self.gen = Generator(self.config)
        self.dis = Discriminator(self.config)
        self.gen_test = copy.deepcopy(self.gen)
        self.criterionNCE = PatchNCELoss()


    def forward(self, co_data, cl_data, mode):
        xa = co_data.cuda()
        xb = cl_data.cuda()
        if mode == 'step_gen':
            c_xa = self.gen.content_encoder(xa)
            s_xa = self.gen.speaker_encoder(xa)
            s_xb = self.gen.speaker_encoder(xb)

            xt = self.gen.decoder(c_xa, s_xb)  # translation
            xr = self.gen.decoder(c_xa, s_xa)  # reconstruction

            l_adv_t = self.dis.calc_gen_loss(xt)
          
            
            c_xt = self.gen.content_encoder(xt)
            c_xr = self.gen.content_encoder(xr)
            l_nce = 0.0
            feat_k = normalize_patches(c_xa)
            feat_q = normalize_patches(c_xt)
            feat_q_ = normalize_patches(c_xr)
            
            for f_q, f_q_, f_k in zip(feat_q, feat_q_, feat_k):
               loss1 = self.criterionNCE(f_q, f_k).mean()
               loss2 = self.criterionNCE(f_q_, f_k).mean()
               l_nce += loss1
               l_nce += loss2
            
            
            l_nce = l_nce / 4 * self.config['lambda']['nce_w']
            l_content = (multi_recon_criterion_l2(c_xt, c_xa) + multi_recon_criterion_l2(c_xr, c_xa)) * self.config['lambda']['c_w']
            # print(l_content)
           
            l_x_rec = self.config['lambda']['r_w'] * recon_criterion(xr, xa)
            l_adv = self.config['lambda']['gan_w'] * l_adv_t
            l_total = (l_adv + l_x_rec  + l_content+ l_nce)
            
            l_total.backward()
            grad_clip([self.gen], self.config['lambda']['max_grad_norm'])
            return l_total, l_x_rec, l_adv, l_content, l_nce
        elif mode == 'step_dis':
            xb.requires_grad_()
            with torch.no_grad():
                c_xa = self.gen.content_encoder(xa)
                s_xb = self.gen.speaker_encoder(xb)
                xt = self.gen.decoder(c_xa, s_xb)
            w_dis, gp = self.dis.cal_dis_loss(xb, xt)
            l_total = - self.config['lambda']['gan_w'] * w_dis + 10 * gp
            l_total.backward()
            grad_clip([self.dis], self.config['lambda']['max_grad_norm'])
            return l_total
        else:
            assert 0, 'Not support operation'

    def inference(self, xa, xb):
        self.gen.eval()
        self.gen_test.eval()
        xa, pad_len = padding_for_inference(xa)
        c_xa_current = self.gen.content_encoder(xa)
        s_xa_current = self.gen.speaker_encoder(xa)
        s_xb_current = self.gen.speaker_encoder(xb)
        xt_current = self.gen.decoder(c_xa_current, s_xb_current)
        xr_current = self.gen.decoder(c_xa_current, s_xa_current)
        c_xa = self.gen_test.content_encoder(xa)
        s_xa = self.gen_test.speaker_encoder(xa)
        s_xb = self.gen_test.speaker_encoder(xb)
        xt = self.gen_test.decoder(c_xa, s_xb)
        xr = self.gen_test.decoder(c_xa, s_xa)
        if pad_len != 0:
            xt = xt[:, :, :-pad_len]
            xr = xr[:, :, :-pad_len]
            xt_current = xt_current[:, :, :-pad_len]
            xr_current = xr_current[:, :, :-pad_len]
        self.gen.train()
        self.gen_test.train()
        return xr_current, xt_current, xr, xt
