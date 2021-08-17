import torch


class addNoise(object):
    def __init__(self, snr):
        self.snr = snr

    def __call__(self, tensor):
        sig_avg_watts = torch.mean(tensor**2)
        sig_avg_db = 10 * torch.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - self.snr
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise = torch.normal(mean_noise, torch.sqrt(noise_avg_watts), tensor.shape)
        # Noise up the original signal
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(snr={self.snr})'