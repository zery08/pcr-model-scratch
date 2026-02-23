import torch

from pcr_ssl.losses import masked_huber_loss, masked_mse_loss


def test_masked_losses_ignore_masked_rows():
    y_pred = torch.tensor([1.0, 3.0])
    y_true = torch.tensor([0.0, 10.0])
    y_mask = torch.tensor([1.0, 0.0])

    mse = masked_mse_loss(y_pred, y_true, y_mask)
    huber = masked_huber_loss(y_pred, y_true, y_mask, delta=1.0)

    assert torch.isclose(mse, torch.tensor(1.0))
    assert torch.isclose(huber, torch.tensor(0.5))
