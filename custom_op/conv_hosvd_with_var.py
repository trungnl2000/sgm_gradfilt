import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn

###### HOSVD base on variance #############

def unfolding(n, A):
    shape = A.shape
    size = th.prod(th.tensor(shape))
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n], lsize)

def truncated_svd(X, var=0.9, driver=None):
    # X is 2D tensor
    U, S, Vt = th.linalg.svd(X, full_matrices=False, driver=driver)
    total_variance = th.sum(S**2)

    explained_variance = th.cumsum(S**2, dim=0) / total_variance
    # k = (explained_variance >= var).nonzero()[0].item() + 1
    nonzero_indices = (explained_variance >= var).nonzero()
    if len(nonzero_indices) > 0:
        # Nếu có ít nhất một phần tử >= var
        k = nonzero_indices[0].item() + 1
    else:
        # Nếu không có phần tử nào >= var, gán k bằng vị trí của phần tử lớn nhất
        k = explained_variance.argmax().item() + 1
    return U[:, :k], S[:k], Vt[:k, :]

def modalsvd(n, A, var, driver):
    nA = unfolding(n, A)
    # return torch.svd(nA)
    return truncated_svd(nA, var, driver)

def hosvd(A, var=0.9, driver=None):
    S = A.clone()
    u_list = []
    for i, ni in enumerate(A.shape):
        u, _, _ = modalsvd(i, A, var, driver)
        S = th.tensordot(S, u, dims=([0], [0]))
        u_list.append(u)
    return S, u_list

def restore_hosvd(S, u_list):
    restored_tensor = S.clone()
    for u in u_list:
        restored_tensor = th.tensordot(restored_tensor, u.t(), dims=([0], [0]))
    return restored_tensor

###############################################################
class Conv2dHOSVDop_with_var(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var, k_hosvd = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)


        S, u_list = hosvd(input, var=var)
        u0, u1, u2, u3 = u_list # B, C, H, W
        if k_hosvd is not None:
            for idx in range(4):
                k_hosvd[idx].append(u_list[idx].shape[1])
            k_hosvd[4].append(input.shape)
        ctx.save_for_backward(S, u0, u1, u2, u3, weight, bias)

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        S, u0, u1, u2, u3, weight, bias  = ctx.saved_tensors
        
        B, C, H, W = u0.shape[0], u1.shape[0], u2.shape[0], u3.shape[0]

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs
        
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input((B,C,H,W), weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            _, _, K_H, K_W = weight.shape
            _, C_prime, H_prime, W_prime = grad_output.shape
            # Pad the input
            u2_padded = pad(u2, (0, 0, padding[0], padding[0]))
            u3_padded = pad(u3, (0, 0, padding[0], padding[0]))

            # Calculate Z1:
            '''
            u0: B, K[0] -> B, K[0], 1, 1, 1
            grad_output: (B, groups*out_channels_per_group+out_channels_per_group, H_prime, W_prime) -> (B, 1, groups*out_channels_per_group+out_channels_per_group, H_prime, W_prime)
            '''
            Z1 = th.sum(u0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * grad_output.unsqueeze(1), dim=0)  # Shape: (K[0], groups*out_channels_per_group+out_channels_per_group, H_prime, W_prime)
            #______________________________________________________________________________________________________________
            # Calculate Z2:
            '''
            S:              shape: K[0], K[1], K[2], K[3]   ->  K[0], K[1], 1, K[2], K[3]
            u2_padded:      shape: H_padded, K[2]           ->  1, 1, H_padded, K[2], 1
            '''
            Z2 = (S.unsqueeze(2) * u2_padded.unsqueeze(0).unsqueeze(1).unsqueeze(-1)).sum(dim=3) # Shape: K[0], K[1], H_padded, K[3]
            #______________________________________________________________________________________________________________
            # Calculate Z3:
            '''
            Z2:         K[0], K[1], H_padded, K[3]  -> K[0], K[1], H_padded, 1, K[3]
            u3_padded:  W_padded, K[3]              -> 1, 1, 1, W_padded, K[3]
            '''
            Z3 = (Z2.unsqueeze(3) * u3_padded.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=4) # Shape: K[0], K[1], H_padded, W_padded
            #______________________________________________________________________________________________________________
            # Calculate Z4
            # Create indices m, k, n, l
            m_indices = th.arange(H_prime, device=Z3.device) * stride[0]
            k_indices = th.arange(K_H, device=Z3.device) * dilation[0]
            # n_indices = th.arange(W_prime) * stride[0]
            # l_indices = th.arange(K_W) * dilation[0]
            # Create grid of m, k, n, l
            m_grid, k_grid = th.meshgrid(m_indices, k_indices, indexing='ij')
            # n_grid, l_grid = th.meshgrid(n_indices, l_indices, indexing='ij',device=Z3.device)

            # Create combination of mk and nl
            combined_indices_m_k = (m_grid + k_grid).t().flatten()
            # combined_indices_n_l = (n_grid + l_grid).t().flatten()
            # Choose Z3
            Z3_selected = Z3[:, :, combined_indices_m_k, :].reshape(Z3.shape[0], Z3.shape[1], K_H, H_prime, Z3.shape[3]) # Shape: K[0], K[1], duyệt qua mk, W_padded -> K[0], K[1], K_H, H_prime, W_padded
            Z3_selected = Z3_selected[:, :, :, :, combined_indices_m_k].reshape(Z3.shape[0], Z3.shape[1], K_H, H_prime, K_W, W_prime) # Shape: K[0], K[1], K_H, H_prime, duyệt qua nl -> K[0], K[1], K_H, H_prime, K_W, W_prime
            # Z3_selected = Z3_selected[:, :, :, :, combined_indices_n_l].reshape(Z3.shape[0], Z3.shape[1], K_H, H_prime, K_W, W_prime) # Shape: K[0], K[1], K_H, H_prime, duyệt qua nl -> K[0], K[1], K_H, H_prime, K_W, W_prime
            # Tính Z4
            '''
            Z3: K[0], K[1], K_H, H_prime, K_W, W_prime  -> K[0], 1, K[1], K_H, H_prime, K_W, W_prime
            Z1: K[0], Cg_prime, H_prime, W_prime        -> K[0], Cg_prime, 1, 1, H_prime, 1, W_prime
            '''
            Z4 = (Z3_selected.unsqueeze(1)*Z1.unsqueeze(2).unsqueeze(3).unsqueeze(5)).sum(dim=0).sum(dim=3).sum(dim=4) # Shape: K[0], Cg_prime, K[1], K_H, H_prime, K_W, W_prime -> Cg_prime, K[1], K_H, K_W,
            
            #______________________________________________________________________________________________________________
            # calculate grad_weight
            if groups == C == C_prime:
                grad_weight = (Z4 * u1.unsqueeze(-1).unsqueeze(-1)).sum(dim=1).unsqueeze(1)
            elif groups == 1:
                Z4_expanded = Z4.unsqueeze(1) #Shape Cg_prime, K[3], K_H, K_W -> Cg_prime, 1, K[3], K_H, K_W
                u1_expanded = u1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  #Shape Cg, K[3] -> 1, Cg, K[3], 1, 1
                grad_weight = (Z4_expanded * u1_expanded).sum(dim=2)
            else: # Havent tensorlize
                print("Havent optimized")


        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class Conv2dHOSVD_with_var(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            var=1,
            k_hosvd = None
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2dHOSVD_with_var, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.var = var
        self.k_hosvd = k_hosvd

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2dHOSVDop_with_var.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var, self.k_hosvd)
        else:
            y = super().forward(x)
        return y

def wrap_convHOSVD_with_var_layer(conv, SVD_var, active, k_hosvd=None):
    new_conv = Conv2dHOSVD_with_var(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         var=SVD_var,
                         k_hosvd = k_hosvd
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv