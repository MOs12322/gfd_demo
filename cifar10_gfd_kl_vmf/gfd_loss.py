from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from cuda_math import bmm as kernel_bmm

_WORKSPACE_BYTES = 512 * 1024**2


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'little') % (2**63-1)


def unit(value):
    return F.normalize(value.double() if value.dtype == torch.float64 else value.float(), dim=-1)


def groups(value, side):
    b, n, tokens, d = value.shape
    grid = math.isqrt(tokens)
    if grid*grid != tokens or side < 1 or grid % side:
        raise ValueError('Group side must divide the square patch grid')
    r = grid // side
    return (value.reshape(b, n, r, side, r, side, d).permute(0, 2, 4, 1, 3, 5, 6)
            .reshape(b*r*r, n, side*side, d)), r*r


@lru_cache(maxsize=32)
def _off_diagonal(n, device):
    indices = torch.arange(n*n)
    return indices[indices.remainder(n+1) != 0].to(device)


def _target_scale(p):
    if p.shape[1] < 2:
        raise ValueError('At least two real references are required')
    m = p.shape[2]
    distances = (m - torch.bmm(p.flatten(2), p.flatten(2).transpose(1, 2))).clamp_min(0)
    distances = distances.flatten(1)[:, _off_diagonal(p.shape[1], str(p.device))]
    return distances.std(-1, unbiased=False).clamp_min(m*1e-6)


def _sample_vmf(x, kappa, seed):
    if x.shape[-1] < 2:
        raise ValueError('Sphere dimension must be at least two')
    rng = torch.Generator(device=x.device).manual_seed(int(seed))
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng)
    direction = unit(noise - (noise*x).sum(-1, keepdim=True)*x)
    k = torch.broadcast_to(kappa.double().reshape(-1, 1, 1), x.shape[:-1])
    valid = (torch.isfinite(k) & (k >= 0)).all()
    if x.is_cuda:
        torch._assert_async(valid, 'Concentration must be finite and nonnegative')
    elif not bool(valid):
        raise ValueError('Concentration must be finite and nonnegative')
    m = float(x.shape[-1]-1)
    b = m / (2*k + torch.sqrt(4*k.square()+m*m))
    x0 = (1-b)/(1+b)
    offset = k*x0 + m*torch.log1p(-x0.square())
    found = torch.zeros_like(k, dtype=torch.bool)
    selected = torch.zeros_like(k)
    alpha = torch.full((4, *k.shape), m/2, device=x.device, dtype=torch.float64)
    for _ in range(32):
        a = torch._standard_gamma(alpha, generator=rng)
        c = torch._standard_gamma(alpha, generator=rng)
        z = a/(a+c).clamp_min(torch.finfo(a.dtype).tiny)
        w = ((1-(1+b)*z)/(1-(1-b)*z)).clamp(-1, 1)
        test = k*w + m*torch.log1p((-x0*w).clamp(min=-1+1e-15)) - offset
        u = torch.rand(test.shape, device=x.device, dtype=test.dtype, generator=rng)
        accepted = test >= u.clamp_min(torch.finfo(u.dtype).tiny).log()
        first = accepted.int().argmax(0)
        valid = accepted.any(0)
        candidate = w.gather(0, first[None])[0]
        selected = torch.where(~found & valid, candidate, selected)
        found |= valid
        if bool(found.all()):
            break
    else:
        raise RuntimeError('Wood proposal budget exhausted')
    radius = (1-selected.square()).clamp_min(0).sqrt().to(x.dtype)
    return unit(selected.to(x.dtype)[..., None]*x + radius[..., None]*direction)


def _weights(query, real, clean, kappa, first=0):
    flat = query.flatten(2)
    p = kernel_bmm(flat, real.flatten(2).transpose(1, 2))*kappa[:, None, None]
    q = kernel_bmm(flat, clean.flatten(2).transpose(1, 2))*kappa[:, None, None]
    rows = torch.arange(first, first+query.shape[1], device=query.device)
    cols = torch.arange(clean.shape[1], device=query.device)
    q.masked_fill_(rows[:, None] == cols[None], -torch.inf)
    return p.softmax(-1), q.softmax(-1)


@torch.no_grad()
def _add_query(total, clean, real, query, kappa, workspace_bytes):
    r, n, m, d = clean.shape
    if n < 2:
        raise ValueError('At least two generated particles are required')

    per_row = clean.element_size()*(6*m*d + 2*(n+real.shape[1]))
    row_budget = max(1, workspace_bytes//per_row)
    rows = min(n, row_budget)
    regions = max(1, min(r, row_budget//rows))
    transport = (_compiled_transport() if clean.is_cuda and clean.dtype == torch.float32
                 else _transport_math)
    for start in range(0, r, regions):
        stop = min(r, start+regions)
        p, x = real[start:stop], clean[start:stop]
        for first in range(0, n, rows):
            last = min(n, first+rows)
            y = query[start:stop, first:last]
            ap, aq = _weights(y, p, x, kappa[start:stop], first)
            delta = (torch.bmm(ap, p.flatten(2)) - torch.bmm(aq, x.flatten(2))).reshape_as(y)
            anchor = x[:, first:last]
            transport(total[start:stop, first:last], anchor, y, delta)


def _transport_math(total, x, y, delta):
    total.add_((x*y).sum(-1, keepdim=True)*delta
               - (x*delta).sum(-1, keepdim=True)*y)


@lru_cache(maxsize=None)
def _compiled_transport():
    return torch.compile(_transport_math, dynamic=True)


def _finish_math(velocity, x, queries):
    velocity = velocity/queries
    return velocity-(velocity*x).sum(-1, keepdim=True)*x


@lru_cache(maxsize=None)
def _compiled_finish():
    return torch.compile(_finish_math, dynamic=False)


def _finish_field(velocity, x, queries):
    fn = _compiled_finish() if x.is_cuda and x.dtype == torch.float32 else _finish_math
    return fn(velocity, x, queries)


def _component_field(x, p, query_center, kappa, queries, seed):
    velocity = torch.zeros_like(x)
    for ell in range(queries):
        y = _sample_vmf(query_center, kappa, seed_for(seed, ell))
        _add_query(velocity, x, p, y, kappa, _WORKSPACE_BYTES)
    return _finish_field(velocity, x, queries)


def _sphere_target(x, velocity, max_angle_deg):
    fn = _compiled_target() if x.is_cuda and x.dtype == torch.float32 else _target_math
    return fn(x, velocity, math.radians(max_angle_deg))


@lru_cache(maxsize=None)
def _compiled_target():
    return torch.compile(_target_math, dynamic=False)


def _target_math(x, velocity, cap):
    v = velocity - (velocity*x).sum(-1, keepdim=True)*x
    length = v.norm(dim=-1, keepdim=True)
    angle = length.clamp(max=cap)
    target = torch.cos(angle)*x + torch.sin(angle)*v/length.clamp_min(1e-12)
    return unit(torch.where(length > 0, target, x))


def _cosine_terms(raw, target):
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    scale = norm.clamp_min(1e-8)
    z = raw/scale
    t = target/torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = (z*t).sum(-1, keepdim=True)
    loss = (1-cosine).clamp_min(0).sum()/raw.shape[0]/raw.shape[1]
    gradient = cosine*z/torch.where(norm>0, norm, torch.ones_like(norm))-t/scale
    gradient = gradient*(cosine<=1)/raw.shape[0]/raw.shape[1]
    return loss, gradient


@lru_cache(maxsize=None)
def _compiled_cosine():
    return torch.compile(_cosine_terms, dynamic=False)


def _ungroups(value, side, batch, tokens):
    grid = math.isqrt(tokens)
    r = grid//side
    return (value.reshape(batch,r,r,value.shape[1],side,side,value.shape[-1])
            .permute(0,3,1,4,2,5,6).reshape(batch,value.shape[1],tokens,value.shape[-1]))


@torch.no_grad()
def _reference_loss_and_gradients(features, real, config, seed):

    total = torch.zeros((), device=next(iter(features.values())).device)
    gradients = {}
    for key in sorted(features):
        leaf = features[key].detach().float()
        gradient = torch.zeros_like(leaf)
        first = unit(leaf)
        field_features = unit(first)
        query_features = unit(field_features)
        real_features = unit(unit(real[key].detach()))
        sides = [1] if leaf.shape[2] == 1 else config['group_sides']
        for side in sides:
            raw, count = groups(leaf.detach(), side)
            x, _ = groups(field_features, side)
            center, _ = groups(query_features, side)
            p, _ = groups(real_features, side)
            scale = _target_scale(p)
            velocity = torch.zeros_like(x)
            for ti, tau in enumerate(config['temperatures']):
                kappa = 1/(tau*scale)
                velocity.add_(_component_field(x, p, center, kappa, config['queries'], seed_for(seed, key, side, ti)))
            velocity.div_(len(config['temperatures']))
            target = _sphere_target(x, velocity, config['max_angle_deg'])
            fn = _compiled_cosine() if raw.is_cuda else _cosine_terms
            value, cotangent = fn(raw, target)
            weight = count/(len(features)*len(sides))
            gradient.add_(_ungroups(cotangent, side, leaf.shape[0], leaf.shape[2]), alpha=weight)
            total.add_(value*weight)
        gradients[key] = gradient
    return total, gradients


@torch.no_grad()
def loss_and_gradients(features, real, config, seed):

    import cuda_field
    if (all(cuda_field.supported(value) and value.shape[1] == 64
            and value.shape[2] in (1, 256) for value in features.values())
            and all(value.shape[1] == 64 for value in real.values())):
        return cuda_field.loss_and_gradients(features, real, config, seed)
    return _reference_loss_and_gradients(features, real, config, seed)
