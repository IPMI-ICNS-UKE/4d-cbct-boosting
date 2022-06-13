ROOSTER4D = {
    'fp': 'CudaRayCast',
    'bp': 'CudaVoxelBased',
    'dimension': (304, 250, 390, 10),
    'spacing': (1.0, 1.0, 1.0, 1.0),
    'origin': (-161, -128, -195, 0),
    # 'dimension': (512, 250, 512, 10),
    # 'spacing': (1.0, 1.0, 1.0, 1.0),
    # 'origin': (0, 0, 0, 0),
    'niter': 10,
    'cgiter': 4,
    'tviter': 10,
    'gamma_time': 0.0002,
    'gamma_space': 0.00005
}

FDK = {
    'hardware': 'cuda',
    'dimension': (304, 250, 390),
    'spacing': (1.0, 1.0, 1.0),
    'origin': (-161, -128, -195),
    'pad': 1.0,
    'hann': 1.0,
    'hannY': 1.0,
    'short': 360
}
