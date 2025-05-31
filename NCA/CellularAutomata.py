import torch
import torch.nn.functional as F
from torch import nn

ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def perchannel_conv(x, filters):
  '''filters: [filter_n, h, w]'''
  b, ch, h, w = x.shape
  y = x.reshape(b*ch, 1, h, w)
  y = F.pad(y, [1, 1, 1, 1], 'circular')
  y = y.to(device)
  filters = filters.to(device)
  y = F.conv2d(y, filters[:,None])
  return y.reshape(b, -1, h, w)

def perception(x):
  filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
  return perchannel_conv(x, filters)

class CA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=96):
    super().__init__()
    self.chn = chn
    self.w1 = nn.Conv2d(chn*4, hidden_n, 1)
    self.w2 = nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()

  def forward(self, x, update_rate=0.5):
    y = perception(x)
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    return x+y*udpate_mask

  def seed(self, n, sz=128):
    return torch.zeros(n, self.chn, sz, sz)

class CPE2D(nn.Module):
  def __init__(self):
    super(CPE2D, self).__init__()
    self.cached_penc = None
    self.last_tensor_shape = None

  def forward(self, tensor):
    if len(tensor.shape) != 4:
      raise RuntimeError("The input tensor has to be 4d!")

    if self.cached_penc is not None and self.last_tensor_shape == tensor.shape:
      return self.cached_penc

    self.cached_penc = None
    batch_size, orig_ch, h, w = tensor.shape
    xs = torch.arange(h, device=tensor.device) / h
    ys = torch.arange(w, device=tensor.device) / w
    xs = 2.0 * (xs - 0.5 + 0.5 / h)
    ys = 2.0 * (ys - 0.5 + 0.5 / w)
    xs = xs[None, :, None]
    ys = ys[None, None, :]
    emb = torch.zeros((2, h, w), device=tensor.device).type(tensor.type())
    emb[:1] = xs
    emb[1: 2] = ys

    self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    self.last_tensor_shape = tensor.shape

    return self.cached_penc

class FullCA(torch.nn.Module):
  def __init__(self, chn=12,c_out=3,fc_dim=96,padding_mode='replicate',pos_emb="CPE",perception_scales=[0, 1, 2, 3, 4, 5], noise_level=0.1, hidden_n=96):
    super().__init__()

    self.c_in = chn
    self.c_out = c_out
    self.perception_scales = perception_scales
    self.fc_dim = fc_dim
    self.padding_mode = padding_mode
    self.pos_emb = pos_emb
    self.expand = 4

    self.c_cond = 0
    if self.pos_emb == 'CPE':
      self.pos_emb_2d = CPE2D()
      self.c_cond += 2
    else:
      self.pos_emb_2d = None

    self.w1 = torch.nn.Conv2d(self.c_in * self.expand + self.c_cond, self.fc_dim, 1)
    torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)

    self.w2 = torch.nn.Conv2d(self.fc_dim, self.c_in, 1, bias=True)
    torch.nn.init.xavier_normal_(self.w2.weight, gain=0.1)
    torch.nn.init.zeros_(self.w2.bias)

    self.sobel_filter_x = torch.FloatTensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    self.sobel_filter_y = self.sobel_filter_x.T

    self.identity_filter = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    self.laplacian_filter = torch.FloatTensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])
    self.register_buffer("noise_level", torch.tensor([noise_level]))

  def perceive_torch(self, x, scale=0):
    assert scale in [0, 1, 2, 3, 4, 5]
    if scale != 0:
      _, _, h, w = x.shape
      h_new = int(h // (2 ** scale))
      w_new = int(w // (2 ** scale))
      x = F.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=False)

    def _perceive_with_torch(z, weight):
      conv_weights = weight.reshape(1, 1, 3, 3).repeat(self.c_in, 1, 1, 1).to("cpu")
      z = F.pad(z, [1, 1, 1, 1], self.padding_mode)
      return F.conv2d(z, conv_weights, groups=self.c_in)

    y1 = _perceive_with_torch(x, self.sobel_filter_x)
    y2 = _perceive_with_torch(x, self.sobel_filter_y)
    y3 = _perceive_with_torch(x, self.laplacian_filter)

    tensor_list = [x]
    tensor_list += [y1, y2, y3]

    y = torch.cat(tensor_list, dim=1)

    if scale != 0:
      y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)

    return y

  def perceive_multiscale(self, x, pos_emb_mat=None):
    perceptions = []
    y = 0
    for scale in self.perception_scales:
      z = self.perceive_torch(x, scale=scale)
      perceptions.append(z)

    y = sum(perceptions)
    y = y / len(self.perception_scales)

    if pos_emb_mat is not None:
      y = torch.cat([y, pos_emb_mat], dim=1)

    return y

  def forward(self, x, update_rate=0.5):

    if self.pos_emb_2d:
      y = self.perceive_multiscale(x, pos_emb_mat=self.pos_emb_2d(x))
    else:
      y = self.perceive_multiscale(x)

    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    return x+y*udpate_mask

  def seed(self, n, sz=128):
    return (torch.rand(n, self.c_in, sz, sz) - 0.5) * self.noise_level

class NoiseCA(torch.nn.Module):
  def __init__(self, chn=12, noise_level=0.1, hidden_n=96):
    super().__init__()
    self.chn = chn
    self.w1 = nn.Conv2d(chn*4, hidden_n, 1)
    self.w2 = nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()
    self.register_buffer("noise_level", torch.tensor([noise_level]))

  def adaptation(self, x):
    x = perception(x)
    return self.w2(torch.relu(self.w1(x)))
  def forward(self, x, update_rate=0.5, rk4Step = False):

    b, c, h, w = x.shape
    udpate_mask = (torch.rand(b, 1, h, w) + update_rate).floor()

    if rk4Step:

      k1 = self.adaptation(x)
      k2 = self.adaptation(x + k1 * 0.5 * udpate_mask)
      k3 = self.adaptation(x + k2 * 0.5 * udpate_mask)
      k4 = self.adaptation(x + k3 * 0.5 * udpate_mask)

      return x + (k1 + 2 * k2 + 2 * k3 + k4) * udpate_mask / 6.0
    else:
      y = perception(x)
      y = self.w2(torch.relu(self.w1(y)))
      b, c, h, w = y.shape
      udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
      return x+y*udpate_mask

  def seed(self, n, sz=128):
    return (torch.rand(n, self.chn, sz, sz) - 0.5) * self.noise_level

def to_rgb(x):
  return x[...,:3,:,:]+0.5