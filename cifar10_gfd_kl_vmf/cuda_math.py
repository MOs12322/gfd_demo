from functools import lru_cache
import torch
try:
    import triton
    import triton.language as tl
    _AVAILABLE = int(triton.__version__.split('.')[0]) >= 3
except ImportError:
    _AVAILABLE = False

if _AVAILABLE:
    @triton.jit
    def _kernel(A, B, C, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
               A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
               B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,
               SPLITS:tl.constexpr,CHUNK:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        tile=tl.program_id(0);batch=tl.program_id(1);split=tl.program_id(2)
        im=(tile//tl.cdiv(N,BN))*BM+tl.arange(0,BM)
        jn=(tile%tl.cdiv(N,BN))*BN+tl.arange(0,BN)
        kk=split*CHUNK+tl.arange(0,BK)
        acc=tl.zeros((BM,BN),tl.float32)
        for _ in range(tl.cdiv(CHUNK,BK)):
            a=tl.load(A+batch*A0+im[:,None]*A1+kk[None,:]*A2,
                      (im[:,None]<M)&(kk[None,:]<K)&(kk[None,:]<(split+1)*CHUNK),0)
            b=tl.load(B+batch*B0+kk[:,None]*B1+jn[None,:]*B2,
                      (kk[:,None]<K)&(kk[:,None]<(split+1)*CHUNK)&(jn[None,:]<N),0)
            acc=tl.dot(a,b,acc,input_precision='tf32x3')
            kk+=BK
        out=C+(batch*SPLITS+split)*M*N+im[:,None]*N+jn[None,:]
        tl.store(out,acc,(im[:,None]<M)&(jn[None,:]<N))


    def _tensor_bmm(a,b):
        batch,m,k=a.shape;n=b.shape[2]
        bm,bn,bk=32,64,32
        blocks=batch*triton.cdiv(m,bm)*triton.cdiv(n,bn)
        splits=min(32,max(1,triton.cdiv(256,blocks)),max(1,k//512))
        chunk=triton.cdiv(triton.cdiv(k,splits),bk)*bk
        partial=torch.empty((batch,splits,m,n),device=a.device,dtype=torch.float32)
        _kernel[(triton.cdiv(m,bm)*triton.cdiv(n,bn),batch,splits)](
            a,b,partial,m,n,k,*a.stride(),*b.stride(),splits,chunk,bm,bn,bk,
            num_warps=4,num_stages=3)
        return partial[:,0] if splits==1 else partial.sum(1)


@lru_cache(maxsize=None)
def _supported(index):
    return torch.version.cuda is not None and torch.cuda.get_device_capability(index)[0] >= 8


def bmm(a, b):
    if (not _AVAILABLE or not a.is_cuda or a.dtype != torch.float32
            or b.dtype != torch.float32 or a.shape[2] < 8192 or b.shape[2] > 128
            or not _supported(a.device.index)):
        return torch.bmm(a, b)
    return _tensor_bmm(a, b)
