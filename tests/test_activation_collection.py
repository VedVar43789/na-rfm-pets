import torch

from na_rfm import ActivationCollector, collect_activations


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.middle_block = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x, t):
        del t
        return self.middle_block(x)


def test_activation_collector_captures_flattened_block_output():
    model = TinyUNet()
    collector = ActivationCollector("middle_block", flatten=True, dtype=torch.float32)
    collector.register(model)
    try:
        _ = model(torch.ones(2, 3, 5, 5), torch.zeros(2, dtype=torch.long))
        activations = collector.capture()
    finally:
        collector.remove()

    assert activations.shape == (2, 4 * 5 * 5)
    assert activations.dtype == torch.float32


def test_collect_activations_returns_labels_and_batch_rows():
    model = TinyUNet()
    images = torch.zeros(3, 3, 5, 5)
    labels = torch.tensor([1, 2, 3])
    dataloader = [(images[:2], labels[:2]), (images[2:], labels[2:])]

    activations, out_labels = collect_activations(
        model,
        dataloader,
        block_name="middle_block",
        alpha_prod_t=0.5,
        timestep=10,
        device="cpu",
        dtype=torch.float32,
    )

    assert activations.shape == (3, 4 * 5 * 5)
    assert torch.equal(out_labels, labels)
