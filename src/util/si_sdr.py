import numpy as np
import torch

def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    """
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    projection = optimal_scaling * reference
    noise = estimation - projection
    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)

def si_sdr_loss(est, ref):
    reference_energy = torch.sum(ref ** 2, dim=-1, keepdim=True)
    print(reference_energy.shape)
    optimal_scaling = torch.sum(ref * est, dim=-1, keepdim=True) / reference_energy
    projection = optimal_scaling * ref
    noise = est - projection
    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1)
    # return 10 * np.log10(ratio)
    return -torch.mean(10 * np.log10(ratio))

if __name__ == '__main__':
    est = torch.rand((5, 1, 16384))
    ref = torch.rand((5, 1, 16384))
    print(si_sdr_loss(est, ref))
    print(si_sdr(est.numpy(), ref.numpy()))