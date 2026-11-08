from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import yaml

from gfd_loss import loss_and_gradients, seed_for
from models import DinoFeatures, build_generator, extract, image_gradient, FEATURE_BATCH

_OBJECTIVE = 'target_only_mean_field_hierarchical_kl'


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    allowed = {'layers', 'group_sides', 'temperatures', 'particles', 'references', 'queries', 'max_angle_deg'}
    if not isinstance(cfg, dict) or set(cfg) != allowed:
        raise ValueError('Configuration must contain exactly the method parameters in config.yaml')
    for key, lower, upper in [('layers', 1, 12), ('group_sides', 1, 16)]:
        values = cfg[key]
        if not values or len(set(values)) != len(values) or any(type(v) is not int or not lower <= v <= upper for v in values):
            raise ValueError(f'Invalid {key}')
    if any(16 % side for side in cfg['group_sides']):
        raise ValueError('Group sides must divide 16')
    if (not cfg['temperatures'] or len(set(cfg['temperatures'])) != len(cfg['temperatures'])
            or any(not math.isfinite(t) or t <= 0 for t in cfg['temperatures'])):
        raise ValueError('Temperatures must be finite and positive')
    for key in ['particles', 'references', 'queries']:
        if type(cfg[key]) is not int or cfg[key] < (1 if key == 'queries' else 2):
            raise ValueError(f'Invalid {key}')
    if not math.isfinite(cfg['max_angle_deg']) or not 0 < cfg['max_angle_deg'] < 180:
        raise ValueError('Invalid maximum angle')
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RealImages:
    def __init__(self, root, seed, dataset=None):
        if dataset is None:
            from torchvision.datasets import CIFAR10
            dataset = CIFAR10(root, train=True, download=False)
        self.images = torch.as_tensor(dataset.data, dtype=torch.uint8)
        labels = torch.as_tensor(dataset.targets)
        self.indices = [(labels == cls).nonzero().flatten() for cls in range(10)]
        if any(not len(ids) for ids in self.indices):
            raise ValueError('All ten CIFAR-10 classes are required')
        self.rng = torch.Generator().manual_seed(seed+1000)

    def sample(self, n):
        labels = torch.randperm(10, generator=self.rng)[:2]
        batches = []
        for cls in labels.tolist():
            pool = self.indices[cls]
            ids = pool[torch.randint(len(pool), (n,), generator=self.rng)]
            batch = self.images[ids].clone()
            flip = torch.rand(n, generator=self.rng) < .5
            batch[flip] = batch[flip].flip(2)
            batches.append(batch)
        return labels, torch.stack(batches)


class Trainer:
    def __init__(self, model, encoder, data, config, device, seed):
        self.model, self.encoder, self.data = model, encoder, data
        self.cfg, self.device, self.seed = copy.deepcopy(config), device, seed
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .95), weight_decay=.01)
        self.noise_rng = torch.Generator(device=device).manual_seed(seed+2000)
        self.step = 0
        import cuda_field
        use_fast = (device.type == 'cuda' and cuda_field.AVAILABLE
                    and torch.cuda.get_device_capability(device)[0] >= 8
                    and config['particles'] == 64 and config['references'] == 64)
        self.implementation = cuda_field.IMPLEMENTATION if use_fast else 'pytorch-reference'
        self.query_rng_version = cuda_field.QUERY_RNG if use_fast else 'torch-wood-v1'

    def update(self):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        c, p = self.cfg['particles'], self.cfg['references']
        labels, raw = self.data.sample(p)
        raw = raw.flatten(0, 1).to(self.device).float().div_(127.5).sub_(1)
        positive = {key: value.reshape(2, p, *value.shape[1:])
                    for key, value in extract(self.encoder, raw).items()}
        del raw
        images = self.model(c=labels.to(self.device).repeat_interleave(c), cfg_scale=1.,
                            deterministic=False, rng=self.noise_rng)['samples']
        features = {key: value.reshape(2, c, *value.shape[1:])
                    for key, value in extract(self.encoder, images).items()}
        loss, gradients = loss_and_gradients(features, positive, self.cfg,
                                             seed_for(self.seed, 'queries', self.step))
        del features, positive
        dx = image_gradient(self.encoder, images, {key: value.flatten(0, 1) for key, value in gradients.items()})
        images.backward(dx.to(images.dtype))
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2., error_if_nonfinite=True)
        loss_value, grad_norm = torch.stack([loss, norm]).float().detach().cpu().tolist()
        if not math.isfinite(loss_value):
            raise FloatingPointError('Nonfinite training loss')
        lr = 1e-6+(2e-4-1e-6)*min((self.step+1)/2000, 1.)
        for group in self.opt.param_groups:
            group['lr'] = lr
        self.opt.step()
        with torch.no_grad():
            for target, current in zip(self.ema.parameters(), self.model.parameters()):
                target.lerp_(current, .001)
            for target, current in zip(self.ema.buffers(), self.model.buffers()):
                target.copy_(current)
        self.step += 1
        return {'step': self.step, 'loss': loss_value, 'grad_norm': grad_norm, 'lr': lr,
                'real_references': p, 'generated_references': c-1, 'queries': self.cfg['queries']}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model': self.model.state_dict(), 'ema': self.ema.state_dict(),
            'optimizer': self.opt.state_dict(), 'step': self.step, 'config': self.cfg,
            'seed': self.seed, 'objective': _OBJECTIVE,
            'field_implementation': self.implementation, 'query_rng_version': self.query_rng_version,
            'feature_batch': FEATURE_BATCH,
            'data_rng': self.data.rng.get_state(), 'noise_rng': self.noise_rng.get_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
        }
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['config'] != self.cfg or state['seed'] != self.seed:
            raise ValueError('Checkpoint configuration or seed differs')
        if state.get('objective') != _OBJECTIVE:
            raise ValueError('Checkpoint objective differs; start a new output directory')
        if (state.get('query_rng_version') != self.query_rng_version
                or state.get('field_implementation') != self.implementation):
            raise ValueError('Checkpoint field implementation or query random stream differs')
        self.model.load_state_dict(state['model'])
        self.ema.load_state_dict(state['ema'])
        self.opt.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.data.rng.set_state(state['data_rng'])
        self.noise_rng.set_state(state['noise_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if self.device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], self.device)

    @torch.no_grad()
    def sample(self, path):
        from torchvision.utils import save_image
        rng = torch.Generator(device=self.device).manual_seed(12345)
        labels = torch.arange(10, device=self.device).repeat_interleave(8)
        images = self.ema(c=labels, cfg_scale=1., deterministic=True, rng=rng)['samples']
        save_image(images.permute(0, 3, 1, 2).float(), path, nrow=8,
                   normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 GFD-KL training')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--dino-weights', required=True, help='Local timm-format ViT-B/16 safetensors')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--out', default='runs/gfd_kl_vmf_final_fast')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', help='Checkpoint path; defaults to OUT/latest.pt when present')
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error('steps must be positive and seed nonnegative')
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and not (out/'latest.pt').exists() and any(out.iterdir()):
        raise ValueError('Output exists without a recoverable checkpoint; choose a new output directory')
    model = build_generator(device)
    encoder = DinoFeatures(args.dino_weights, config['layers'], device)
    trainer = Trainer(model, encoder, RealImages(args.data_root, args.seed), config,
                      device, args.seed)
    resume = Path(args.resume) if args.resume else out/'latest.pt'
    if args.resume or resume.exists():
        trainer.restore(resume)
    if trainer.step > args.steps:
        raise ValueError('Requested end step precedes the restored step')
    metrics = out/'metrics.jsonl'
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if any(row['step'] > trainer.step for row in rows):
            metrics.rename(out/f'metrics.interrupted.{time.time_ns()}.jsonl')
            metrics.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['step'] <= trainer.step))
    while trainer.step < args.steps:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        row = trainer.update()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        row['seconds'] = time.perf_counter()-started
        if trainer.step == 1 or trainer.step % 20 == 0 or trainer.step == args.steps:
            with (out/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if trainer.step % 1000 == 0 or trainer.step == args.steps:
            trainer.sample(out/f'samples_{trainer.step:07d}.png')
        if trainer.step % 5000 == 0 or trainer.step == args.steps:
            state = out/f'state_{trainer.step:07d}.pt'
            trainer.save(state)
            link = out/'latest.tmp'
            if link.exists():
                link.unlink()
            try:
                os.link(state, link)
            except OSError:
                shutil.copyfile(state, link)
            os.replace(link, out/'latest.pt')
    print(f'Finished {trainer.step} updates: {out}', flush=True)


if __name__ == '__main__':
    main()
