ROOSTER4D = {
    "fp": "CudaRayCast",
    "bp": "CudaVoxelBased",
    "dimension": (384, 150, 384, 10),
    "spacing": (1.0, 1.0, 1.0, 1.0),
    "origin": (-384 / 2, -150 / 2, -384 / 2, 0),
    "n_iter": 10,
    "cg_iter": 4,
    "tv_iter": 10,
    "gamma_time": 0.0002,
    "gamma_space": 0.00005,
}

FDK = {
    "hardware": "cuda",
    "dimension": (384, 150, 384),
    "spacing": (1.0, 1.0, 1.0),
    "origin": (-384 / 2, -150 / 2, -384 / 2),
    "pad": 1.0,
    "hann": 1.0,
    "hann_y": 1.0,
    "short": 360,
}
