import torch


def si_sdr_loss():
    def _loss_function(ref, est):
        ref = ref - torch.mean(ref)
        est = est - torch.mean(est)
        reference_energy = torch.sum(ref ** 2, dim=-1, keepdim=True)
        optimal_scaling = torch.sum(ref * est, dim=-1, keepdim=True) / reference_energy
        projection = optimal_scaling * ref
        noise = est - projection
        ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1)
        # return 10 * np.log10(ratio)
        return -torch.mean(10 * torch.log10(ratio))

    return _loss_function


def multi_scale_si_sdr_loss(weights):
    def _loss_function(ref, short_est, middle_est, long_est):
        assert len(weights) == 3
        short_si_sdr_loss = si_sdr_loss()(ref, short_est)
        middle_si_sdr_loss = si_sdr_loss()(ref, middle_est)
        long_si_sdr_loss = si_sdr_loss()(ref, long_est)
        loss = weights[0] * short_si_sdr_loss + weights[1] * middle_si_sdr_loss + weights[2] * long_si_sdr_loss
        return loss, (short_si_sdr_loss, middle_si_sdr_loss, long_si_sdr_loss)

    return _loss_function


if __name__ == '__main__':
    est = torch.rand((5, 1, 16384))
    ref = torch.rand((5, 1, 16384))
    print(multi_scale_si_sdr_loss([0.1, 0.2, 0.7])(est, ref, ref, ref))
