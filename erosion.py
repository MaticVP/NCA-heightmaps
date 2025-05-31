import time
import argparse

import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from noise import pnoise2
matplotlib.use('Agg')
def simple_gradient(a):
    dx = 0.5 * (np.roll(a, 1, axis=0) - np.roll(a, -1, axis=0))
    dy = 0.5 * (np.roll(a, 1, axis=1) - np.roll(a, -1, axis=1))
    return 1j * dx + dy

def gaussian_blur(a, sigma=1.0):
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    kernel = g(freq_radial)
    kernel /= kernel.sum()
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real

def normalize(x, bounds=(0, 1)):
    return np.interp(x, (x.min(), x.max()), bounds)

def apply_slippage(terrain, repose_slope, cell_width):
    delta = simple_gradient(terrain) / cell_width
    smoothed = gaussian_blur(terrain, sigma=1.5)
    result = np.select([np.abs(delta) > repose_slope], [smoothed], terrain)
    return result

def fbm(shape, p, lower=-np.inf, upper=np.inf):
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    envelope = (np.power(freq_radial, p, where=freq_radial!=0) *
                (freq_radial > lower) * (freq_radial < upper))
    envelope[0][0] = 0.0
    phase_noise = np.exp(2j * np.pi * np.random.rand(*shape))
    return normalize(np.real(np.fft.ifft2(np.fft.fft2(phase_noise) * envelope)))

def perl_noise(shape):
    noise_2d = np.zeros(shape)
    scale = 20.0
    height, width = shape

    for i in range(height):
        for j in range(width):
            x = i / scale
            y = j / scale
            noise_2d[i][j] = pnoise2(x, y, octaves=4, persistence=0.5, lacunarity=2.0)

    noise_2d -= noise_2d.min()
    noise_2d /= noise_2d.max()
    return noise_2d

def lerp(x, y, a): return (1.0 - a) * x + a * y

def displace(a, delta):
    fns = {-1: lambda x: -x, 0: lambda x: 1 - np.abs(x), 1: lambda x: x}
    result = np.zeros_like(a)
    for dx in range(-1, 2):
        wx = np.maximum(fns[dx](delta.real), 0.0)
        for dy in range(-1, 2):
            wy = np.maximum(fns[dy](delta.imag), 0.0)
            result += np.roll(np.roll(wx * wy * a, dy, axis=0), dx, axis=1)
    return result

def sample(a, offset):
    shape = np.array(a.shape)
    delta = np.array((offset.real, offset.imag))
    coords = np.array(np.meshgrid(*map(range, shape))) - delta

    lower_coords = np.floor(coords).astype(int)
    upper_coords = lower_coords + 1
    coord_offsets = coords - lower_coords
    lower_coords %= shape[:, np.newaxis, np.newaxis]
    upper_coords %= shape[:, np.newaxis, np.newaxis]

    result = lerp(lerp(a[lower_coords[1], lower_coords[0]],
                       a[lower_coords[1], upper_coords[0]],
                       coord_offsets[0]),
                  lerp(a[upper_coords[1], lower_coords[0]],
                       a[upper_coords[1], upper_coords[0]],
                       coord_offsets[0]),
                  coord_offsets[1])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procedural erosion simulation")
    parser.add_argument('--noise', type=str, choices=['fbm', 'perlin'], required=True,
                        help="Noise type to use: 'fbm' or 'perlin'")
    parser.add_argument('--outfile', type=str, required=True, help="Output filename (e.g., './maps/terrain.png')")
    parser.add_argument('--dim', type=int, default=256, help="Output image dimension (e.g., 256, 512)")
    args = parser.parse_args()

    dim = args.dim
    shape = [dim, dim]
    full_width = 200
    cell_width = full_width / dim
    cell_area = cell_width ** 2

    rain_rate = 0.0008 * cell_area
    evaporation_rate = 0.0005
    min_height_delta = 0.05
    repose_slope = 0.03
    gravity = 30.0
    gradient_sigma = 0.5
    sediment_capacity_constant = 50.0
    dissolving_rate = 0.25
    deposition_rate = 0.001
    iterations = dim

    start = time.time()
    terrain = fbm(shape, -2.0) if args.noise == 'fbm' else perl_noise(shape)

    sediment = np.zeros_like(terrain)
    water = np.zeros_like(terrain)
    velocity = np.zeros_like(terrain)

    for i in range(iterations):
        print('%d / %d' % (i + 1, iterations))
        water += np.random.rand(*shape) * rain_rate

        gradient = simple_gradient(terrain)
        gradient = np.select([np.abs(gradient) < 1e-10],
                             [np.exp(2j * np.pi * np.random.rand(*shape))],
                             gradient)
        gradient /= np.abs(gradient)

        neighbor_height = sample(terrain, -gradient)
        height_delta = terrain - neighbor_height

        sediment_capacity = (
            (np.maximum(height_delta, min_height_delta) / cell_width) * velocity *
            water * sediment_capacity_constant)

        deposited_sediment = np.select(
            [height_delta < 0, sediment > sediment_capacity],
            [np.minimum(height_delta, sediment),
             deposition_rate * (sediment - sediment_capacity)],
            default=dissolving_rate * (sediment - sediment_capacity))

        deposited_sediment = np.maximum(-height_delta, deposited_sediment)

        sediment -= deposited_sediment
        terrain += deposited_sediment
        sediment = displace(sediment, gradient)
        water = displace(water, gradient)
        terrain = apply_slippage(terrain, repose_slope, cell_width)

        velocity = gravity * height_delta / cell_width
        water *= 1 - evaporation_rate

    elapsed = time.time() - start
    print(f"Heightmap generated in {elapsed:.2f} seconds.")

    image = Image.fromarray(np.round(terrain * 255).astype('uint8'))
    image.save(args.outfile)
