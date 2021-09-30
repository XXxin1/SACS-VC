import torch.nn.functional as F
from math import ceil
from tensorboardX import SummaryWriter
import torch
import yaml
from data.vctk_dataset import *
import soundfile as sf
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

def smooth_l1_loss(input, target, beta=0.10, reduction = 'mean'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


def recon_criterion_l2(predict, target):
    target = target.detach()
    return torch.mean((predict - target) ** 2)


def multi_recon_criterion_l2(predict, target):
    loss = 0.0
    for i in range(0, len(predict)):
        loss += recon_criterion_l2(predict[i], target[i])
    return loss / len(predict)


def padding_for_inference(inp):
    pad_len = 0
    while ceil(inp.shape[-1] / 4) % 2 != 0 or ceil(inp.shape[-1] / 2) % 2 != 0 or ceil(inp.shape[-1]) % 2 != 0:
        inp = F.pad(inp, [0, 1], 'reflect')
        pad_len += 1
    return inp, pad_len


def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


def save_config(config, args):
    with open(f'{args.store_model_path}.config.yaml', 'w') as f:
        yaml.dump(config, f)
    with open(f'{args.store_model_path}.args.yaml', 'w') as f:
        yaml.dump(vars(args), f)
    return


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)


def get_data_loaders(config, args):
    data_dir = args.data_dir
    train_dataset = VCTKDateset(os.path.join(data_dir, args.train_set))
    in_test_dataset = VCTKDateset_name(os.path.join(data_dir, args.seen_set))
    out_test_dataset = VCTKDateset_name(os.path.join(data_dir, args.unseen_set))

    train_dataloader = DataLoader(train_dataset, batch_size=config['data_loader']['batch_size'], shuffle=True,
                                  num_workers=4, drop_last=True, pin_memory=True,
                                  collate_fn=VCTK_collate)
    in_test_dataloader = DataLoader(in_test_dataset, batch_size=1, shuffle=True,
                                    num_workers=1)
    out_test_dataloader = DataLoader(out_test_dataset, batch_size=1, shuffle=True,
                                     num_workers=1)

    return infinite_iter(train_dataloader), infinite_iter(in_test_dataloader), infinite_iter(out_test_dataloader)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def write_waveform(wav_data, iteration, mode, store_wav_path, file_name, sample_rate, logger):
    os.makedirs(os.path.join(store_wav_path, str(iteration), mode), exist_ok=True)
    sf.write(os.path.join(store_wav_path, str(iteration), mode, file_name), wav_data, sample_rate)
    logger.audio_summary(f'{mode}/{file_name}', wav_data, iteration, sample_rate)
    return


def random_sample_patches(feats, num_patches=128, patch_ids=None):
    
    return_ids = []
    return_feats = []
    for feat_id, feat in enumerate(feats):
        if patch_ids is not None:
            patch_id = patch_ids[feat_id]
        else:
            patch_id = torch.randperm(feat.shape[2], device=feats[0].device)
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
        x_sample = feat[:, :, patch_id]
        return_ids.append(patch_id)
        return_feats.append(F.normalize(x_sample, dim=1, p=2))
        
    return return_feats, return_ids 
        
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out 
        

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
           return 1
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

    def audio_summary(self, tag, value, step, sr):
        self.writer.add_audio(tag, value, step, sample_rate=sr)
