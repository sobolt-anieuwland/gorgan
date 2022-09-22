from typing import Dict, Union, List, Tuple, Any

from PIL.Image import Image


class Renderer:
    """Super class for rendering in the GAN framework."""

    def __init__(self):
        """Initializes the renderer."""

    def __call__(
        self, batch: Dict[str, Any], batch_idx: int, renamings: Dict[str, str] = {}
    ) -> Dict[str, Image]:
        """Convert a training + inference batch to a dictionary of panels.

        Parameters
        ----------
        batch: Dict[str, Any]
            A batch of data sources as dictionary to render.
        batch_idx: int
            The index within  a batch.
        renamings:Dict[str, str]
            Mapping of batch name to renderer name.
        Returns
        -------
        Dict[str, Image]
            A training batch of data sources as a dictionary of rendered panels
        """
        raise NotImplementedError
