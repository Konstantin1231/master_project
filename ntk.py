"""
The core of NTK realization was taking from the pytorch "ntk" tutorial. Repository link : https://github.com/pytorch/tutorials/blob/main/intermediate_source/neural_tangent_kernels.py
"""
import torch
from torch.func import functional_call, vmap, vjp, jvp, jacrev
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full', show_dim_jac = False):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]
    if show_dim_jac:
        i = 0
        for jac_1_i in jac1:
            print(f"Layer {list(params.keys())[i]} has jacobian shape:  {jac_1_i.shape}")
            i+=1
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]
    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False

    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return [result[:, 0].detach().numpy()[0], jac1]

def empirical_ntk_ntk_vps(func, params, x1, x2, compute='full'):
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes ``vec @ J(x2).T``
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes ``J(X1) @ vjps``
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    # ``get_ntk(x1, x2)`` computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to ``empirical_ntk_ntk_vps`` are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the ``vmaps`` here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)

    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    if compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)

