from functools import lru_cache
import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as lib
    from triton.language.random import philox
    AVAILABLE = int(triton.__version__.split('.')[0]) >= 3
except ImportError:
    AVAILABLE = False

IMPLEMENTATION = 'fused-field-v1'
QUERY_RNG = 'philox-wood-v1'
WORKSPACE_BYTES = 512 * 1024**2


if AVAILABLE:
    @triton.jit
    def _uniform_pair(seed, index, draw, stream):
        a, b, c, d = philox(seed, index.to(tl.uint32), (index >> 32).to(tl.uint32),
                            (tl.full((), 0, tl.uint32) + draw).to(tl.uint32),
                            (tl.full((), 0, tl.uint32) + stream).to(tl.uint32))

        u = ((a.to(tl.uint64) << 21) | (b.to(tl.uint64) >> 11)).to(tl.float64)
        v = ((c.to(tl.uint64) << 21) | (d.to(tl.uint64) >> 11)).to(tl.float64)
        return (u + 1.) * (1. / 9007199254740992.), v * (1. / 9007199254740992.)

    @triton.jit
    def _gamma(seed, index, stream, alpha: tl.constexpr, live):

        shape: tl.constexpr = alpha + 1. if alpha < 1. else alpha
        d: tl.constexpr = shape - 1. / 3.
        c = 1. / lib.sqrt(tl.full((), 9. * d, tl.float64))
        found = ~live
        value = tl.full(index.shape, 1., tl.float64)
        draw = tl.full((), 0, tl.int32)
        while (tl.sum((~found).to(tl.int32), 0) > 0) & (draw < 256):
            u1, u2 = _uniform_pair(seed, index, draw * 2, stream)
            normal = lib.sqrt(-2. * lib.log(u1)) * lib.cos(6.283185307179586 * u2)
            u, _ = _uniform_pair(seed, index, draw * 2 + 1, stream)
            base = 1. + c * normal
            cube = base * base * base
            square = normal * normal
            accepted = (base > 0.) & ((u < 1. - .0331 * square * square) |
                (lib.log(u) < .5 * square + d * (1. - cube + lib.log(tl.maximum(cube, 1.e-300)))))
            value = tl.where(~found & accepted, d * cube, value)
            found = found | accepted
            draw += 1
        if alpha < 1.:
            u, _ = _uniform_pair(seed, index, 1024, stream)
            value *= lib.exp(lib.log(u) / alpha)
        return value, found

    @triton.jit
    def _radial_kernel(K, Seeds, W, Error, N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                       S: tl.constexpr, T: tl.constexpr, R: tl.constexpr,
                       REGION_START, QUERY_START,
                       L: tl.constexpr, B: tl.constexpr):
        ids = tl.program_id(0) * B + tl.arange(0, B)
        valid = ids < R * S * N * M
        factor = ids % M
        particle = (ids // M) % N
        query = (ids // (M * N)) % S + QUERY_START
        region = ids // (M * N * S)
        index = ((region.to(tl.uint64) + REGION_START) * N + particle) * M + factor
        seed = tl.load(Seeds + query, mask=valid, other=0)
        k = tl.load(K + region * T + query // L, mask=valid, other=0.).to(tl.float64)
        good = (k >= 0.) & (k < float('inf'))
        if tl.sum((valid & ~good).to(tl.int32), 0) > 0:
            tl.atomic_max(Error, 1)
        m: tl.constexpr = float(D - 1)
        b = m / (2. * k + lib.sqrt(4. * k * k + m * m))
        x0 = (1. - b) / (1. + b)
        offset = k * x0 + m * lib.log1p(-x0 * x0)
        found = ~valid | ~good
        selected = tl.full((B,), 0., tl.float64)
        attempt = tl.full((), 0, tl.int32)
        while (tl.sum((~found).to(tl.int32), 0) > 0) & (attempt < 128):
            a, ok_a = _gamma(seed, index, attempt * 3 + 1, m / 2., ~found)
            c, ok_c = _gamma(seed, index, attempt * 3 + 2, m / 2., ~found)
            if tl.sum((~ok_a | ~ok_c).to(tl.int32), 0) > 0:
                tl.atomic_max(Error, 2)
            z = a / tl.maximum(a + c, 1.e-300)
            w = tl.minimum(1., tl.maximum(-1., (1. - (1. + b) * z) / (1. - (1. - b) * z)))
            score = k * w + m * lib.log1p(tl.maximum(-x0 * w, -1. + 1.e-15)) - offset
            u, _ = _uniform_pair(seed, index, 0, attempt * 3 + 3)
            accepted = (score >= lib.log(u)) & ok_a & ok_c
            selected = tl.where(~found & accepted, w, selected)
            found = found | accepted
            attempt += 1
        if tl.sum((~found).to(tl.int32), 0) > 0:
            tl.atomic_max(Error, 3)
        tl.store(W + ids, selected, mask=valid)

    @triton.jit
    def _query_kernel(X, Seeds, W, Y, N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                      S: tl.constexpr, REGION_START,
                      QUERY_START, B: tl.constexpr):
        row = tl.program_id(0)
        factor = row % M
        particle = (row // M) % N
        query = (row // (M * N)) % S + QUERY_START
        region = row // (M * N * S)
        index = ((region.to(tl.uint64) + REGION_START) * N + particle) * M + factor
        seed = tl.load(Seeds + query)
        pair = tl.arange(0, B // 2)
        u1, u2 = _uniform_pair(seed, index, pair, 0)

        radius = lib.sqrt(-2. * lib.log(u1.to(tl.float32)))
        angle = 6.283185307179586 * u2.to(tl.float32)
        noise = tl.reshape(tl.join(radius * lib.cos(angle), radius * lib.sin(angle)), (B,))
        channel = tl.arange(0, B)
        x = tl.load(X + ((region * N + particle) * M + factor) * D + channel, channel < D, 0.)
        noise = tl.where(channel < D, noise, 0.)
        tangent = noise - tl.sum(noise * x, 0) * x
        tangent = tangent / tl.maximum(lib.sqrt(tl.sum(tangent * tangent, 0)), 1.e-12)
        w = tl.load(W + row)
        radial = lib.sqrt(tl.maximum(0., 1. - w * w)).to(tl.float32)
        y = w.to(tl.float32) * x + radial * tangent
        y = y / tl.maximum(lib.sqrt(tl.sum(y * y, 0)), 1.e-12)
        tl.store(Y + row * D + channel, y, channel < D)

    @triton.jit
    def _weight_kernel(Logits, K, A, N: tl.constexpr, P: tl.constexpr, Q: tl.constexpr,
                       T: tl.constexpr, L: tl.constexpr, QUERY_START: tl.constexpr,
                       B: tl.constexpr):
        row, region = tl.program_id(0), tl.program_id(1)
        j = tl.arange(0, B)
        stride: tl.constexpr = N + P
        k = tl.load(K + region * T + (row // N + QUERY_START) // L)
        lp = tl.load(Logits + (region * Q + row) * stride + j, j < P, -float('inf')) * k
        lq = tl.load(Logits + (region * Q + row) * stride + P + j, j < N, -float('inf')) * k
        lq = tl.where(j == row % N, -float('inf'), lq)
        ep = tl.exp(lp - tl.max(lp, 0)); eq = tl.exp(lq - tl.max(lq, 0))
        ap = ep / tl.sum(ep, 0); aq = eq / tl.sum(eq, 0)
        tl.store(A + (region * Q + row) * stride + j, ap, j < P)
        tl.store(A + (region * Q + row) * stride + P + j, -aq, j < N)

    @triton.jit
    def _centroid_transport(A, Z, X, Y, V, N: tl.constexpr, P: tl.constexpr,
                            Q: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                            BM: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
        block, factor, region = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rows = block * BM + tl.arange(0, BM)
        channels = tl.arange(0, BD)
        refs = tl.arange(0, BK)
        x = tl.load(X + ((region * N + rows[:, None]) * M + factor) * D + channels[None, :],
                    (rows[:, None] < N) & (channels[None, :] < D), 0.)
        result = tl.zeros((BM, BD), tl.float32)
        for query in range(Q // N):
            query_rows = query * N + rows
            acc = tl.zeros((BM, BD), tl.float32)
            for start in range(tl.cdiv(N + P, BK)):
                j = start * BK + refs
                a = tl.load(A + (region * Q + query_rows[:, None]) * (N + P) + j[None, :],
                            (rows[:, None] < N) & (j[None, :] < N + P), 0.)
                z = tl.load(Z + ((region * (N + P) + j[:, None]) * M + factor) * D + channels[None, :],
                            (j[:, None] < N + P) & (channels[None, :] < D), 0.)
                acc = tl.dot(a, z, acc, input_precision='tf32x3')
            y = tl.load(Y + ((region * Q + query_rows[:, None]) * M + factor) * D + channels[None, :],
                        (rows[:, None] < N) & (channels[None, :] < D), 0.)
            result += tl.sum(x * y, 1)[:, None] * acc - tl.sum(x * acc, 1)[:, None] * y
        ptr = V + ((region * N + rows[:, None]) * M + factor) * D + channels[None, :]
        mask = (rows[:, None] < N) & (channels[None, :] < D)
        tl.store(ptr, tl.load(ptr, mask, 0.) + result, mask)


def supported(value):
    return (AVAILABLE and value.is_cuda and value.dtype in (torch.float32, torch.bfloat16)
            and value.shape[-1] == 768 and torch.cuda.get_device_capability(value.device)[0] >= 8)


def sample_queries(center, concentrations, seeds, queries, *, region_start=0, query_start=0,
                   query_count=None, error=None):

    if not center.is_cuda or center.dtype != torch.float32 or center.shape[-1] < 2:
        raise ValueError('CUDA query centers must be FP32 with at least two channels')
    center = center.contiguous()
    concentrations = concentrations.contiguous()
    r, n, m, d = center.shape
    t = concentrations.shape[1]
    s = t * queries if query_count is None else query_count
    if queries < 1 or s < 1 or query_start < 0 or query_start+s > t*queries or seeds.numel() != t*queries:
        raise ValueError('Invalid query count or query seed array')
    if error is None:
        error = torch.zeros((), device=center.device, dtype=torch.int32)
        check = True
    else:
        check = False
    w = torch.empty((r, s*n, m), device=center.device, dtype=torch.float64)
    y = torch.empty((r, s*n, m, d), device=center.device, dtype=torch.float32)
    _radial_kernel[(triton.cdiv(w.numel(), 128),)](
        concentrations, seeds, w, error, n, m, d, s, t, r, region_start, query_start, queries, 128)
    _query_kernel[(w.numel(),)](center, seeds, w, y, n, m, d, s, region_start, query_start,
                                triton.next_power_of_2(d))
    if check:
        torch._assert_async(error == 0, 'Exact query sampler failed: invalid concentration or rejection budget exhausted')
    return y


def layer_scales(p, sides):

    from gfd_loss import _off_diagonal, _target_scale, groups
    from cuda_math import _tensor_bmm
    c, n, tokens, d = p.shape
    grid = math.isqrt(tokens)
    per = p.permute(0, 2, 1, 3).contiguous()
    flat = per.reshape(c*tokens, n, d)
    gram = _tensor_bmm(flat, flat.transpose(1, 2)).reshape(c, grid, grid, n, n)
    result = {}
    for side in sides:


        if side >= 8:
            result[side] = _target_scale(groups(p, side)[0])
            continue
        r = grid // side
        joint = gram.reshape(c, r, side, r, side, n, n).sum((2, 4)).reshape(c*r*r, n, n)
        distances = (side*side-joint).clamp_min(0).flatten(1)[:, _off_diagonal(n, str(p.device))]
        result[side] = distances.std(-1, unbiased=False).clamp_min(side*side*1e-6)
    return result


@lru_cache(maxsize=None)
def _finish():
    from gfd_loss import _finish_math
    def fn(v, x, n):
        return _finish_math(v, x, n)
    return torch.compile(fn, dynamic=True)


def combined_field(x, p, center, kappa, seeds, queries, error, workspace_bytes=WORKSPACE_BYTES,
                   supplied=None):
    from cuda_math import bmm
    r, n, m, d = x.shape
    np = p.shape[1]
    total_queries = kappa.shape[1] * queries
    velocity = torch.zeros_like(x)

    one = 4*m*d
    per_query = n*one + 8*n*m + 8*n*(n+np)
    base = (2*n+np)*one
    s = max(1, min(total_queries, (workspace_bytes-base)//per_query))
    regions = max(1, min(r, workspace_bytes//(base+s*per_query)))
    regions = 2**int(math.log2(regions))
    for start in range(0, r, regions):
        end = min(r, start+regions)
        xx, pp, cc = x[start:end], p[start:end], center[start:end]
        z = torch.cat([pp, xx], dim=1).flatten(2)
        kk = kappa[start:end].contiguous()
        for first in range(0, total_queries, s):
            count = min(s, total_queries-first)
            if supplied is None:
                y = sample_queries(cc, kk, seeds, queries, region_start=start, query_start=first,
                                   query_count=count, error=error)
            else:
                y = supplied[start:end, first:first+count].reshape(end-start, count*n, m, d).contiguous()
            logits = bmm(y.flatten(2), z.transpose(1, 2))
            a = torch.empty_like(logits)
            _weight_kernel[(count*n, end-start)](logits, kk, a, n, np, count*n, kappa.shape[1],
                                                queries, first, triton.next_power_of_2(max(n, np)))
            _centroid_transport[(triton.cdiv(n, 16), m, end-start)](
                a, z, xx, y, velocity[start:end], n, np, count*n, m, d, 16, triton.next_power_of_2(d), 16,
                num_warps=8, num_stages=1)
    return _finish()(velocity, x, total_queries)


@lru_cache(maxsize=None)
def _regression():
    from gfd_loss import _target_math, _cosine_terms
    def fn(raw, x, velocity, cap):
        return _cosine_terms(raw, _target_math(x, velocity, cap))
    return torch.compile(fn, dynamic=False)


@torch.no_grad()
def loss_and_gradients(features, real, config, seed):
    from gfd_loss import unit, groups, _ungroups, seed_for
    device = next(iter(features.values())).device
    total = torch.zeros((), device=device)
    gradients = {}
    error = torch.zeros((), device=device, dtype=torch.int32)
    for key in sorted(features):
        leaf = features[key].detach().float()
        gradient = torch.zeros_like(leaf)
        field_features = unit(unit(leaf))
        query_features = unit(field_features)
        real_features = unit(unit(real[key].detach()))
        sides = [1] if leaf.shape[2] == 1 else config['group_sides']
        scales = layer_scales(real_features, sides)
        for side in sides:
            raw, count = groups(leaf, side)
            x, _ = groups(field_features, side)
            center, _ = groups(query_features, side)
            p, _ = groups(real_features, side)
            kappa = torch.stack([1/(tau*scales[side]) for tau in config['temperatures']], dim=1)
            seeds = torch.tensor([seed_for(seed_for(seed, key, side, ti), ell)
                                  for ti in range(len(config['temperatures'])) for ell in range(config['queries'])],
                                 dtype=torch.int64).pin_memory().to(device, non_blocking=True)
            velocity = combined_field(x, p, center, kappa, seeds, config['queries'], error)
            value, cotangent = _regression()(raw, x, velocity, math.radians(config['max_angle_deg']))
            weight = count/(len(features)*len(sides))
            gradient.add_(_ungroups(cotangent, side, leaf.shape[0], leaf.shape[2]), alpha=weight)
            total.add_(value*weight)
        gradients[key] = gradient
    torch._assert_async(error == 0, 'Exact query sampler failed: invalid concentration or rejection budget exhausted')
    return total, gradients
