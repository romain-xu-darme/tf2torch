import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
import torchaudio.functional as func


# Copying: https://github.com/kahst/BirdNET-Analyzer/issues/177
class MelSpecLayerSimple(Module):
    """The MelSpecLayerSimple module is a copy of the spectrogram generator from BirdNet."""

    def __init__(
        self,
        sample_rate: int = 48000,
        spec_shape: int = 96,
        frame_step: int = 278,
        frame_length: int = 2048,
        fmin: int = 0,
        fmax: int = 3000,
        filter_bank: None | Tensor | list[list[float]] | np.ndarray = None,
        **kwargs,
    ):
        """
        Initializes the layer.  Default values are the ones in BirdNET .yml model file from V2.4.
        :param sample_rate: the sample rate used for input tensors.
        :param spec_shape: the number of frequencies in the spectrogram.  Its first item should be the number of frequencies of interest.
        :param frame_step: the step of a frame, used to compute sliding windows.
        :param frame_length: the length of a frame.
        :param fmin: the minimal frequency of interest.
        :param fmax: the maximal frequency of interest.
        :param filter_bank: an optional argument (shape (spec_shape, frame_length//2 +1)
          that indicates the filterbank
          used to translate the spectrogram into a MEL spectrogram.
          If `None`, the spectrogram is calculated using `torchaudio.functional.melscale_fbanks`
          and the parameters given as input.
        :param kwargs: other arguments.
        """
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        # self.data_format = data_format
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax

        # A mel_filterbank is a linear matrix that converts non-mel spectrograms into mel spectrograms
        if filter_bank is None:
            self.mel_filterbank = func.melscale_fbanks(  # Input parameters:
                n_freqs=self.frame_length // 2 + 1,
                sample_rate=self.sample_rate,  # Output parameters
                n_mels=self.spec_shape[0],
                f_min=self.fmin,
                f_max=self.fmax,
            )  # Shape (NON_MEL_FREQS, MEL_FREQS) where NON_MEL_FREQS = 513 and MEL_FREQS = 96
        elif isinstance(filter_bank, torch.Tensor):
            self.mel_filterbank = filter_bank
        elif isinstance(filter_bank, np.ndarray):
            self.mel_filterbank = torch.Tensor(filter_bank)
        elif isinstance(filter_bank, list):
            self.mel_filterbank = torch.Tensor(np.array(filter_bank))
        else:
            raise ValueError("Expected 'None', a torch tensor, an array or a numpy array.")
        self.mag_scale = 1.23

    @staticmethod
    def _normalize(tensor: Tensor, dim: int) -> Tensor:
        # Normalize values between -1 and 1
        epsilon = 0.000001
        tensor = tensor - torch.min(tensor, dim=dim, keepdim=True, out=None).values
        tensor = tensor / (torch.max(tensor, dim=dim, keepdim=True).values + epsilon)
        tensor = tensor - 0.5
        tensor = tensor * 2
        return tensor
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.call(inputs)

    def call(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, list):
            inputs = inputs[0]

        # inputs, shape (B, SIZE_WINDOW)
        inputs = MelSpecLayerSimple._normalize(tensor=inputs, dim=1)

        # Perform STFT
        spec = torch.stft(
            inputs,
            win_length=self.frame_length,
            n_fft=self.frame_length,
            # Should be 2**x such that >= win_length
            hop_length=self.frame_step,
            normalized=False,
            center=False,
            window=torch.hann_window(self.frame_length).to(inputs),
            pad_mode="reflect",
            return_complex=True,
            # `return_complex=False` is being deprecated
        )  # shape (B, NON_MEL_FREQS, N) of complex64 where N = SIZE_WINDOW // HOP_LENGTH

        # Cast from complex to float
        # BirdNet implementation was using `spec.to(torch.float32)` which was printing a warning.
        spec = torch.view_as_real(spec)[:, :, :, 0]  # shape (B, NON_MEL_FREQS, N) of float32

        # Apply mel scale
        melspec = torch.tensordot(spec, self.mel_filterbank, [[1], [0]])  # Shape (B, N, MEL_FREQS)

        # Convert to power spectrogram
        melspec = torch.pow(melspec, 2.0)

        # Convert magnitudes using nonlinearity
        melspec = torch.pow(melspec, 1.0 / (1.0 + torch.math.exp(self.mag_scale)))
        melspec = torch.flip(melspec, dims=[2])

        melspec = torch.swapdims(melspec, 1, 2)  # Shape (B, MEL_FREQS, N)
        # melspec = melspec.reshape(melspec.shape + (1,))  # Shape (B, MEL_FREQS, N, 1)
        melspec = melspec.reshape((melspec.shape[0], 1,) + melspec.shape[1:])  # Shape (B, 1, MEL_FREQS, N)

        return melspec

    def nb_frequencies(self) -> int:
        return self.spec_shape

    def expected_sample_rate(self) -> int:
        return self.sample_rate
