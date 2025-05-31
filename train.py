import random
import argparse

import PIL.Image
import cv2
import numpy as np
import pylab as pl
import torch.optim
from cv2 import imread
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn

from NCA.CellularAutomata import CA, to_rgb, NoiseCA, FullCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

vgg16 = models.vgg16(weights='IMAGENET1K_V1').features


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1)*255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)


def calc_styles_vgg(imgs):
    style_layers = [1, 6, 11, 18, 25]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (imgs - mean) / std
    b, c, h, w = x.shape
    features = [x.reshape(b, c, h * w)]
    for i, layer in enumerate(vgg16[:max(style_layers) + 1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            features.append(x.reshape(b, c, h * w))
    return features


def project_sort(x, proj):
    return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]


def ot_loss(source, target, proj_n=32):
    ch, n = source.shape[-2:]
    projs = F.normalize(torch.randn(ch, proj_n), dim=0)
    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)
    target_interp = F.interpolate(target_proj, n, mode='nearest')
    return (source_proj - target_interp).square().sum()


def create_vgg_loss(target_img):
    yy = calc_styles_vgg(target_img)

    def loss_f(imgs):
        xx = calc_styles_vgg(imgs)
        return sum(ot_loss(x, y) for x, y in zip(xx, yy))

    return loss_f


def to_nchw(img):
    img = torch.as_tensor(img)
    if len(img.shape) == 3:
        img = img[None, ...]
    return img.permute(0, 3, 1, 2)


def seed_with_random_chunks(ca, style_img, res, chunk_size=32, num_chunks=3):
    x = ca.seed(1, res)

    style_tensor = to_nchw(style_img)[0, :3]

    for _ in range(num_chunks):
        h0 = random.randint(0, res - chunk_size)
        w0 = random.randint(0, res - chunk_size)
        x[0, :3, h0:h0 + chunk_size, w0:w0 + chunk_size] = style_tensor[:, h0:h0 + chunk_size, w0:w0 + chunk_size]

    return x


def train(style_img, ca_type, res, use_chunks=False, lr=1e-3, numSteps=2000, save_path="ca_model.pt"):
    if ca_type == "VCA":
        ca = CA()
    elif ca_type == "NCA":
        ca = NoiseCA(noise_level=0.2)
    elif ca_type == "FCA":
        ca = FullCA(noise_level=0.2)
    else:
        raise ValueError(f"Unknown CA type: {ca_type}")

    opt = torch.optim.Adam(ca.parameters(), lr, capturable=True)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [numSteps // 2, numSteps], 0.3)
    loss_log = []

    with torch.no_grad():
        if use_chunks:
            pool = ca.seed(256, res)
            chunk_size = np.random.randint(8, 64)
            chunk_num = np.random.randint(2, 15)
            pool = torch.cat([
                seed_with_random_chunks(ca, style_img, res=res, chunk_size=chunk_size, num_chunks=chunk_num)
                for _ in range(256)
            ], dim=0)
        else:
            pool = ca.seed(256, res)

        loss_f = create_vgg_loss(to_nchw(style_img))

    gradient_checkpoints = False  # Set True if OOM problems

    for i in range(numSteps):
        with torch.no_grad():
            batch_idx = np.random.choice(len(pool), 4, replace=False)
            x = pool[batch_idx]
            if i % 8 == 0:
                if not use_chunks:
                    x[:1] = ca.seed(1, res)
                else:
                    chunk_size = np.random.randint(8, 64)
                    chunk_num = np.random.randint(2, 15)
                    x[:1] = seed_with_random_chunks(ca, style_img, res=res, chunk_size=chunk_size, num_chunks=chunk_num)

        step_n = np.random.randint(32, 96)
        if not gradient_checkpoints:
            for k in range(step_n):
                x = ca(x)
        else:
            x.requires_grad = True
            x = torch.utils.checkpoint.checkpoint_sequential([ca] * step_n, 16, x)

        overflow_loss = (x - x.clamp(-1.0, 1.0)).abs().sum()
        loss = loss_f(to_rgb(x)) + overflow_loss
        with torch.no_grad():
            loss.backward()
            for p in ca.parameters():
                p.grad /= (p.grad.norm() + 1e-8)
            opt.step()
            opt.zero_grad()
            lr_sched.step()
            pool[batch_idx] = x

            loss_log.append(loss.item())
            if i % 5 == 0:
                print(f'''
            Step: {i}
            Loss: {loss.item():.6f}
            LR: {lr_sched.get_last_lr()[0]:.6f}''')

            if i % 50 == 0:
                pl.plot(loss_log, '.', alpha=0.1)
                pl.yscale('log')
                pl.ylim(np.min(loss_log), loss_log[0])
                pl.tight_layout()
                pl.show()
                imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
                pl.imshow(np.hstack(imgs))
                torch.save(ca.state_dict(), save_path)
                print(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Cellular Automata style model.")
    parser.add_argument('--resolution', type=int, default=128, help='Resolution of the input image and CA (e.g. 64, 128, 256)')
    parser.add_argument('--type', type=str, default="NCA", choices=["VCA", "NCA", "FCA"], help="Type of Cellular Automata model")
    parser.add_argument('--use_chunks', action='store_true', help="Seed with random chunks")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training steps (epochs)')
    parser.add_argument('--save_path', type=str, default="ca_model.pt", help="File path to save the trained model")
    parser.add_argument('--style_img_path', type=str, required=True, help="Path to the ground truth/style image")

    args = parser.parse_args()

    style_img = imread(args.style_img_path)
    image = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.resolution, args.resolution))
    image = image / 255.0
    image = np.float32(image)
    image = torch.from_numpy(image).to(device)

    train(image, args.type, args.resolution, args.use_chunks, lr=args.lr, numSteps=args.epochs, save_path=args.save_path)

if __name__ == "__main__":
    main()
