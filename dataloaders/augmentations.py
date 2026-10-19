import numpy as np
from scipy import signal
import torch


class DegradeToSNR:
    """Target SNR is passed per-call, not at init, since we iterate over a list of targets."""

    def __call__(
        self,
        tensor: torch.Tensor,
        noise_floor_db: float,
        pulse_power_db: float,
        target_snr_db: float
    ) -> torch.Tensor:

        # No SNR degradation requested.
        # Return immediately before doing any SNR calculations.
        if target_snr_db == "NA":
            return tensor, None

        pulse_power = 10 ** (pulse_power_db / 10)
        noise_power = 10 ** (noise_floor_db / 10)

        signal_power = pulse_power - noise_power

        if signal_power <= 0:
            raise ValueError(
                f"Invalid PDW values: "
                f"pulse_power_db={pulse_power_db}, "
                f"noise_floor_db={noise_floor_db}, "
                f"signal_power={signal_power}"
            )

        current_snr_db = 10 * np.log10(signal_power / noise_power)

        if current_snr_db <= target_snr_db:
            return tensor, current_snr_db

        target_SNR = 10 ** (target_snr_db / 10)

        additive_noise_power = (signal_power / target_SNR) - noise_power

        if additive_noise_power < 0:
            raise ValueError(
                f"Negative additive noise power: {additive_noise_power}"
            )

        x = tensor.numpy().astype(np.float32).copy()

        noise_I = np.random.normal(
            0,
            np.sqrt(additive_noise_power / 2),
            len(tensor)
        )
    
        noise_Q = np.random.normal(
            0,
            np.sqrt(additive_noise_power / 2),
            len(tensor)
        )

        x[:, 0] += noise_I
        x[:, 1] += noise_Q

        return torch.from_numpy(x), target_snr_db


class FrequencyShift:
    """
    Applies a SPECIFIC frequency shift to a complex baseband I/Q signal --
    deterministic mechanism, the shift amount is provided by the caller.
    Used both for random augmentation (via RandomFrequencyShift) and for
    deliberate baseband CFO correction using a real PDW-measured value.

    Input:  (length, 2) - [I, Q]
    Output: (length, 2) - [I, Q], shifted by shift_hz
    """
    def __call__(self, tensor: torch.Tensor, shift_hz: float, fs: float) -> torch.Tensor:
        I = tensor[:, 0].numpy()
        Q = tensor[:, 1].numpy()
        signal = I + 1j * Q

        N = signal.shape[0]
        t = np.arange(N, dtype=np.float64) / fs
        osc = np.exp(1j * 2 * np.pi * shift_hz * t)

        shifted = signal * osc

        result = np.stack([shifted.real, shifted.imag], axis=-1).astype(np.float32)
        return torch.from_numpy(result)


class RandomFrequencyShift:
    """
    Wraps FrequencyShift: samples
    shift_hz ~ Uniform(offset_hz - max_shift_hz, offset_hz + max_shift_hz)
    per call. Draws from NumPy's global RNG (same stream as DegradeToSNR).

    offset_hz defaults to 0.0, reproducing the original symmetric
    Uniform(-max_shift_hz, +max_shift_hz) range exactly. Setting offset_hz
    re-centres that same width elsewhere -- e.g. max_shift_hz=100e3 with
    offset_hz=100e3 gives Uniform(0, 200e3) instead of Uniform(-100e3, 100e3).
    """
    def __init__(self, max_shift_hz: float, offset_hz: float = 0.0):
        self.max_shift_hz = max_shift_hz
        self.offset_hz = offset_hz
        self.frequency_shift = FrequencyShift()

    def __call__(self, tensor: torch.Tensor, fs: float) -> torch.Tensor:
        shift_hz = np.random.uniform(
            self.offset_hz - self.max_shift_hz, self.offset_hz + self.max_shift_hz
        )
        return self.frequency_shift(tensor, shift_hz=shift_hz, fs=fs)
        
class PhaseShift:
    """
    Applies a SPECIFIC phase rotation to a complex baseband I/Q signal --
    deterministic mechanism, the angle is provided by the caller. Used
    both for random augmentation (via RandomPhaseShift) and for deliberate
    phase correction, mirroring how FrequencyShift/RandomFrequencyShift
    split mechanism from random-augmentation policy.

    Input:  (length, 2) - [I, Q]
    Output: (length, 2) - [I, Q], rotated by theta:
        I' + jQ' = (I + jQ) * e^{j*theta}
    which expands to:
        I' = I*cos(theta) - Q*sin(theta)
        Q' = I*sin(theta) + Q*cos(theta)
    """
    def __call__(self, tensor: torch.Tensor, theta: float) -> torch.Tensor:
        tensor = tensor.float()
        I = tensor[:, 0]
        Q = tensor[:, 1]

        cos_t = torch.cos(torch.tensor(theta, dtype=tensor.dtype))
        sin_t = torch.sin(torch.tensor(theta, dtype=tensor.dtype))

        I_rot = I * cos_t - Q * sin_t
        Q_rot = I * sin_t + Q * cos_t

        return torch.stack([I_rot, Q_rot], dim=1)

class RandomPhaseShift:
    """
    Wraps PhaseShift: samples theta ~ Uniform(0, 2*pi) per call. Draws
    from NumPy's global RNG (same stream as DegradeToSNR), so a single
    np.random.seed(...) call controls both, rather than phase shift
    living on its own separate, independently-seeded RNG stream.

    Useful as an augmentation since a receiver's carrier phase offset is
    arbitrary - the model shouldn't be sensitive to it.

    Input:  (length, 2) - [I, Q]
    Output: (length, 2) - [I, Q], rotated by a random phase
    """
    def __init__(self):
        self.phase_shift = PhaseShift()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        theta = np.random.uniform(0, 2 * np.pi)
        return self.phase_shift(tensor, theta=theta)
        
class AddMagnitude:
    """
    Input:  (length, 2) - [I, Q]
    Output: (3, length) - [I, Q, magnitude]
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:

        # Convert integer IQ samples to float before arithmetic
        tensor = tensor.float()

        I = tensor[:, 0]
        Q = tensor[:, 1]

        mag = torch.sqrt(I**2 + Q**2)

        return torch.stack([I, Q, mag], dim=0)

class NormaliseByPeakMagnitude:
    """
    Input:  (3, length)  - [I, Q, mag]; the peak is read from the mag channel
            or any tensor, if `peak` is passed explicitly (e.g. I/Q only, no mag channel)
    Output: input / peak
    """

    def __call__(self, tensor: torch.Tensor, peak=None) -> torch.Tensor:
        if peak is None:
            peak = tensor[2].abs().max()

        if not torch.isfinite(peak):
            raise ValueError(f"Non-finite peak magnitude: {peak}")

        if peak <= 1e-12:
            raise ValueError("Zero/near-zero magnitude pulse encountered")

        return tensor / peak
        
class Flatten:
    """
    Flattens a multi-dimensional tensor into a single 1D tensor. Useful
    for classical methods (PCA, kNN, SVM, etc.) that expect one flat
    feature vector per sample, rather than a shaped (channels, length)
    tensor.

    Input:  any shape, e.g. (3, L) [I, Q, mag] or (L, 2) [I, Q]
    Output: (numel,) -- flattened in the tensor's existing memory order
    (row-major / C-order), equivalent to tensor.reshape(-1).

    Note: PCA doesn't care about feature ORDER (permuting flattened
    features just permutes rows/columns of the covariance matrix, which
    doesn't change the eigenvalues or resulting components at all) -- so
    which axis gets flattened first here is a free choice, not something
    that needs careful tuning.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1)
        


