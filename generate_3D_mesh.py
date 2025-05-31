import argparse
import numpy as np
import trimesh
from PIL import Image

def gen_map(img, resolution, pixel_scale, height_scale):
    resize_shape = resolution
    pixel_size = pixel_scale
    height_scale = height_scale

    img = img.resize(resize_shape, resample=Image.BILINEAR)
    height_map = np.array(img).astype(np.float32)

    smoothed_map = np.clip(height_map, np.percentile(height_map, 5), np.percentile(height_map, 95))
    smoothed_map -= smoothed_map.min()
    smoothed_map /= smoothed_map.max()
    z_flat = (height_map.flatten()) * height_scale

    h, w = height_map.shape
    x = np.arange(w) * pixel_size
    y = np.arange(h) * pixel_size
    x_grid, y_grid = np.meshgrid(x, y)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    vertices = np.vstack((x_flat, y_flat, z_flat)).T
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = i * w + j
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces)

    face_colors = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        height_value = np.mean(height_map[np.divmod(face, w)])
        norm_value = (height_value - height_map.min()) / (height_map.max() - height_map.min())
        if norm_value > 0.8:
            face_colors[i] = [0.902, 0.882, 0.843]
        elif norm_value > 0.4:
            face_colors[i] = [0.78, 0.678, 0.416]
        elif norm_value > 0.15:
            face_colors[i] = [0.58, 0.878, 0.588]
        else:
            face_colors[i] = [0.459, 0.725, 0.839]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)

    return mesh, face_colors

def normalize_mesh(mesh):
    bounds = mesh.bounds  # (min, max)
    size = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    max_dim = size.max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / max_dim)

def main():
    parser = argparse.ArgumentParser(description="Generate a colored heightmap mesh from an image")
    parser.add_argument('--image', type=str, required=True, help="Input image path (grayscale preferred)")
    parser.add_argument('--resolution', nargs=2, type=int, default=[128, 128], metavar=('W', 'H'), help="Output mesh resolution (width height)")
    parser.add_argument('--pixel_scale', type=float, default=1.0, help="Horizontal spacing between points")
    parser.add_argument('--height_scale', type=float, default=10.0, help="Vertical height scale")
    parser.add_argument('--output', type=str, default='output.obj', help="Output OBJ file path")
    parser.add_argument('--visualize', action='store_true', help="Show interactive visualization before export")
    args = parser.parse_args()

    img = Image.open(args.image).convert('L')

    mesh, face_colors = gen_map(img, tuple(args.resolution), args.pixel_scale, args.height_scale)

    rot = trimesh.transformations.rotation_matrix(
        angle=np.radians(-90),
        direction=[1, 0, 0],
        point=[0, 0, 0]
    )

    mesh.apply_transform(rot)
    normalize_mesh(mesh)

    if args.visualize:
        mesh.show()

    mesh.export(args.output)
    print(f"Mesh exported to {args.output}")

if __name__ == "__main__":
    main()
