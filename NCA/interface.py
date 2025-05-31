import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from NCA.CellularAutomata import CA, to_rgb, NoiseCA, FullCA

def gen_hightmap(type, start_seed=None,numSteps=10,steps=12, res=64, seed_size=8):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    if type == "Perlin" or type == "FBM":
        ca = CA()
    elif type == "Noise FBM" or type == "Noise Perlin" or type == "Chunk Perlin" or type == "Chunk FBM":
        ca = NoiseCA(noise_level=0.2)
    else:
        ca = FullCA(noise_level=0.2)

    if type == "Perlin":
        ca.load_state_dict(torch.load("./Weights/ca_model_pearl_ero.pt", weights_only=True, map_location=torch.device(device)))

    if type == "FBM":
        ca.load_state_dict(torch.load("./Weights/ca_model_fbm_ero.pt", weights_only=True, map_location=torch.device(device)))

    if type == "Noise Perlin":
        ca.load_state_dict(torch.load("./Weights/nca_model_perlin_ero.pt", weights_only=True, map_location=torch.device(device)))

    if type == "Noise FBM":
        ca.load_state_dict(torch.load("./Weights/nca_model_fbm_ero.pt", weights_only=True, map_location=torch.device(device)))

    if type == "Full Perlin":
        ca.load_state_dict(torch.load("./Weights/fca_model_perlin_ero.pt", weights_only=True, map_location=torch.device(device)))

    if type == "Full FBM":
        ca.load_state_dict(torch.load("./Weights/fca_model_fbm_ero.pt", weights_only=True, map_location=torch.device(device)))

    if type == "Chunk Perlin":
        ca.load_state_dict(torch.load("./Weights/fca_model_perlin_ero_chunks.pt", weights_only=True, map_location=torch.device(device)))

    if type == "Chunk FBM":
        ca.load_state_dict(torch.load("./Weights/fca_model_fbm_ero_chunks.pt", weights_only=True, map_location=torch.device(device)))


    if start_seed is None:
        x = ca.seed(1, res)
    else:
        x = ca.seed(1, res)
        ref_x, ref_y, ref_xy = start_seed

        def apply_ref(ref, h_slice, w_slice):
            if ref is not None:
                ref = torch.from_numpy(ref.transpose(2, 0, 1))
                x[0, :3, h_slice, w_slice] = ref[:, h_slice, w_slice]

        chunk_size = seed_size

        apply_ref(ref_x, slice(0, res), slice(0, chunk_size))

        apply_ref(ref_y, slice(0, chunk_size), slice(0, res))

        apply_ref(ref_xy, slice(0, chunk_size), slice(0, chunk_size))

    ca = ca.to(device)
    x = x.to(device)

    with torch.no_grad():
        for i in range(numSteps):

            for k in range(steps):

                x[:] = ca(x)

        #x = to_rgb(x).permute([0, 2, 3, 1]).cpu()
        x = to_rgb(x).permute([0, 2, 3, 1])

        x = x.squeeze(0)

        x = x.detach().cpu().numpy()

        x = np.clip(x, 0, 1)

        return x
