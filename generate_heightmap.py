import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import time
from NCA.interface import gen_hightmap
import argparse
matplotlib.use('Agg')

METHOD_CHOICES = [
    "FBM",
    "Noise Perlin",
    "Noise FBM",
    "Full Perlin",
    "Full FBM",
    "Chunk Perlin",
    "Chunk FBM"
]

def main(grid_size, method, res, num_steps, steps, seed_size, filename):
    map = [None for _ in range(grid_size * grid_size)]

    for i in range(grid_size):
        seed = [None, None, None]
        for j in range(grid_size):
            if i > 0:
                seed[1] = map[(i - 1) * grid_size + j]
            if j > 0 and i > 0:
                index = (i - 1) * grid_size + (j - 1)
                seed[2] = map[index]

            now = time.time()
            uploaded_file = gen_hightmap(
                method, seed, res=res, numSteps=num_steps, steps=steps, seed_size=seed_size
            )
            delta = time.time() - now
            print(f"Generated tile ({i},{j}) in {delta:.2f} seconds")
            seed[0] = uploaded_file
            map[i * grid_size + j] = uploaded_file

    rows = []
    for i in range(grid_size):
        row = [map[i * grid_size + j] for j in range(grid_size)]
        row_combined = np.hstack(row)
        rows.append(row_combined)

    full_image = np.vstack(rows)

    plt.axis('off')
    plt.imshow(full_image, cmap='gray')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tiled heightmaps using NCA.")
    parser.add_argument("--grid_size", type=int, default=1, help="Size of the grid (default: 1)")
    parser.add_argument("--method", type=str, choices=METHOD_CHOICES, default="Full FBM",
                        help=f"Method for generation (choices: {', '.join(METHOD_CHOICES)})")
    parser.add_argument("--res", type=int, default=128, help="Resolution of each tile (default: 128)")
    parser.add_argument("--num_steps", type=int, default=14, help="Number of update steps (default: 14)")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps (default: 5)")
    parser.add_argument("--seed_size", type=int, default=32, help="Seed size (default: 32)")
    parser.add_argument("--filename", type=str, default="output.png", help="Output filename (default: output.png)")

    args = parser.parse_args()

    main(
        grid_size=args.grid_size,
        method=args.method,
        res=args.res,
        num_steps=args.num_steps,
        steps=args.steps,
        seed_size=args.seed_size,
        filename=args.filename
    )
