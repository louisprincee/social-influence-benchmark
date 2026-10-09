import torch
import geoopt
from geoopt.manifolds.lorentz import Lorentz


def Lorentzian_vector(x, gamma=1.0):
    x_0 = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + gamma)
    x = torch.cat([x_0, x], dim=-1)
    return x


def Lorentzian_inner_product(x, y, keepdim=False):
    xy = x * y
    if xy.dim() == 3:
        xy[:, :, 0] = - xy[:, :, 0]
    elif xy.dim() == 2:
        xy[:, 0] = - xy[:, 0]
    else:
        raise ValueError("tensor dim should be 2 or 3.")
    L_product = torch.sum(xy, dim=-1, keepdim=keepdim)
    return L_product


def Lorentzian_distance(x, y, keepdim=False):
    L_product = Lorentzian_inner_product(x, y, keepdim=keepdim)
    L_distance = torch.arccosh(-L_product)
    return L_distance


def Lorentz_to_Klein(x):
    x_0 = x[:, 0].unsqueeze(1)
    x = x[:, 1:] / x_0
    return x


def Lorentz_to_Poincare(x):
    x_0 = x[:, 0].unsqueeze(1) + 1
    x = x[:, 1:] / x_0
    return x


def Lorentz_centroid(x, gamma=1.0, nu=None):
    if nu is not None:
        x = x * nu
    if x.dim() == 2:
        v = x.sum(dim=0, keepdim=True)
    elif x.dim() == 3:
        v = x.sum(dim=1, keepdim=True)
    else:
        raise ValueError("tensor dim should be 2 or 3.")
    v_L_norm = Lorentzian_inner_product(v, v, keepdim=True)
    gamma = torch.tensor(gamma)
    miu = torch.sqrt(gamma) / torch.sqrt(torch.abs(v_L_norm)) * v
    return miu


def Centroid(x, y, gamma, weights=None):
    if isinstance(gamma, torch.Tensor) is False:
        gamma = torch.tensor(gamma)
    if weights is not None:
        a = weights[0] * x + weights[1] * y
    else:
        a = x + y
    modulus_a = torch.sqrt(Lorentzian_inner_product(a, a, keepdim=True).abs())
    mu = torch.sqrt(gamma) * a / modulus_a
    return mu


def Centroid_group(x, gamma, weights=None):
    if isinstance(gamma, torch.Tensor) is False:
        gamma = torch.tensor(gamma)
    if weights is not None:
        if isinstance(weights, torch.Tensor) is False:
            weights = torch.tensor(weights).unsqueeze(1).to(x.device)
        x = x * weights
        a = torch.sum(x, dim=0, keepdim=True)
    else:
        a = torch.sum(x, dim=0, keepdim=True)
    modulus_a = torch.sqrt(Lorentzian_inner_product(a, a, keepdim=True).abs())
    mu = torch.sqrt(gamma) * a / modulus_a
    return mu


def squared_lorentzian_distance(x, y, gamma=1.0):
    xy = Lorentzian_inner_product(x, y)
    sL_distance = -2 * gamma - 2 * xy
    return sL_distance


def squared_lorentzian_distance_from_euc(context_vecs, target_vecs):
    """ Calculation of similarity score via Lorentz squared distance. """
    c = 1.0
    x = context_vecs  # b*L*d
    y = target_vecs  # b*t*d
    x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c)  # b*L*1
    y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c)  # b*t*1
    u = torch.cat((x, x2_srt), dim=-1)  # (b,L, d+1)
    v = torch.cat((y, y2_srt), dim=-1)  # (b,t, d+1)
    # u,v have different shape[0]
    uv = u @ v.T
    result = - 2 * c - 2 * uv
    result = result.neg()

    return result


def position_adjustment(x, gamma=1.0):
    manifold = Lorentz(gamma)
    c = torch.tensor(manifold.k.item())
    # vectors (x) should be Lorentzian vector.
    centroid = Centroid_group(x, gamma=c).expand(x.shape[0], -1)
    origin = torch.zeros_like(x)
    origin[:, 0] = torch.sqrt(c)
    u = manifold.logmap(origin, x)
    u = manifold.transp(origin, centroid, u)
    u = manifold.expmap(centroid, u)

    return u


if __name__ == "__main__":
    c = 1.0
    man = Lorentz(c)
    B = 3
    N = 10
    dim = 5
    x = torch.randn(B, dim)
    x = man.projx(x)
    y = torch.randn(5)
    dis = squared_lorentzian_distance(y, x)
    print(dis)
    y = y.unsqueeze(0)
    dis = squared_lorentzian_distance(y, x)
    print(dis)
    # res = position_adjustment(x, man)

    # print(res)
