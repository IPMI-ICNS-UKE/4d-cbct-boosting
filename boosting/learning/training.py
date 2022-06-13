import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from boosting.learning.datasets import CBCTBoostingDataset
from boosting.learning.losses import L2Loss


def train_one_epoch(
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    data_loader: DataLoader,
    device: str = "cuda",
):
    epoch_losses = []
    model.train(True)
    for data in data_loader:
        pseudo_average_image = data["pseudo_average_image"]
        average_image = data["average_image"]
        # shape (1, n_phases, x, y) for pseudo average image
        # shape (1, x, y) for average image

        pseudo_average_image = pseudo_average_image.swapaxes(0, 1)
        average_image = average_image.unsqueeze(1)
        average_image = torch.repeat_interleave(
            average_image, repeats=pseudo_average_image.shape[0], dim=0
        )

        pseudo_average_image = torch.as_tensor(
            pseudo_average_image, dtype=torch.float32, device=device
        )
        average_image = torch.as_tensor(
            average_image, dtype=torch.float32, device=device
        )

        optimizer.zero_grad()
        prediction = model(pseudo_average_image)
        loss = loss_function(prediction, average_image)
        loss.requires_grad_(True)
        loss.backward()

        optimizer.step()
        loss = loss.detach().cpu().numpy()
        epoch_losses.append(loss)

    return epoch_losses


if __name__ == "__main__":
    from boosting.learning.models import ResidualDenseNet2D
    import boosting.learning.blocks as blocks
    from boosting.learning.blocks import ConvReLU2D
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    DEVICE = "cuda:0"
    CLIM = (0, 1)

    dataset = CBCTBoostingDataset.load("/datalake/4d_cbct_lmu/Hamburg/dataset.pkl")

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = ResidualDenseNet2D(
        in_channels=1,
        out_channels=1,
        growth_rate=16,
        n_blocks=2,
        n_block_layers=4,
        convolution_block=blocks.ConvInstanceNormReLU2D,
        # convolution_block=ConvReLU2D,
        local_feature_fusion_channels=32,
        alpha=0.9,
        pre_block_channels=32,
        post_block_channels=32,
    ).to(DEVICE)
    optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad])
    # loss_function = nn.MSELoss()
    loss_function = L2Loss(gradient_attention=0.5)
    # loss_function = nn.L1Loss()

    for i in range(10):
        epoch_losses = train_one_epoch(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            data_loader=data_loader,
            device=DEVICE,
        )
        print(i, np.mean(epoch_losses))

    d = dataset[150]

    fig, ax = plt.subplots(2, 10, sharex=True, sharey=True)
    fig.suptitle("pavg vs avg")
    for n in range(10):
        ax[0, n].imshow(d["pseudo_average_image"][n], clim=CLIM)
        ax[1, n].imshow(d["average_image"], clim=CLIM)

    prediction = model(
        torch.as_tensor(
            d["pseudo_average_image"][:, np.newaxis], dtype=torch.float32, device=DEVICE
        )
    )
    prediction = prediction.detach().cpu().numpy()
    prediction = prediction.squeeze()


    fig, ax = plt.subplots(2, 11, sharex=True, sharey=True)
    fig.suptitle("pavg vs prediction")
    for n in range(10):
        ax[0, n].imshow(d["pseudo_average_image"][n], clim=CLIM)
        ax[1, n].imshow(prediction[n], clim=CLIM)
    ax[0, 10].imshow(d["average_image"], clim=CLIM)
    ax[1, 10].imshow(d["average_image"], clim=CLIM)

    test_image = sitk.ReadImage("/datalake/4d_cbct_lmu/Hamburg/CBCT4D.mhd")
    test_image_arr = sitk.GetArrayFromImage(test_image)[:, np.newaxis]

    percentiles = np.percentile(test_image_arr, (1, 99))
    test_image_arr = dataset.preprocess(test_image_arr, input_value_range=percentiles, output_value_range=(0, 1))

    prediction = np.zeros_like(test_image_arr)
    for i in range(test_image_arr.shape[-2]):
        image_slice = test_image_arr[..., i, :]
        image_slice = torch.as_tensor(image_slice, dtype=torch.float32, device=DEVICE)

        p = model(image_slice)
        p = p.detach().cpu().numpy()
        prediction[..., i, :] = p

    prediction = dataset.preprocess(prediction, input_value_range=(0, 1), output_value_range=percentiles)
    prediction = prediction.squeeze()
    prediction = sitk.GetImageFromArray(prediction, isVector=False)
    prediction.CopyInformation(test_image)
    sitk.WriteImage(prediction, "/datalake/4d_cbct_lmu/Hamburg/CBCT4D_boosted.mha")
