"""
/src/emgonset/utils/io.py
"""

from ..utils.internals import public_api


@public_api
class EMGTransformCompose:
    """
    Compose multiple EMG transforms together while properly handling initialization.
    Similar to torchvision.transforms.Compose but with initialize() support.
    """

    def __init__(self, transforms):
        """
        Args:
            transforms: List of transforms to apply sequentially
        """
        self.transforms = transforms

    def initialize(self, fs: float) -> None:
        """Initialize all contained transforms with sampling frequency"""
        for transform in self.transforms:
            if hasattr(transform, "initialize"):
                transform.initialize(fs)

    def __call__(self, tensor):
        """Apply all transforms sequentially"""
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor
