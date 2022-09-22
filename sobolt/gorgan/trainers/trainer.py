from typing import Optional, Dict, Any

from sobolt.gorgan import Dataset


class Trainer:
    """ Interface for the trainers. Trainers are classes implementing the main training
        loop. Given a training set, and maybe a validation set, they take care of the
        actual training proces.
    """

    def __init__(self, config: Dict[str, Any] = {}, quiet: bool = False):
        pass

    def train(self, train_set: Dataset, val_set: Optional[Dataset], init_epoch: int = 1):
        """ Trains the GAN.

            Parameters
            ----------
            train_set: In1FlexDataset
                The data set to train on. Its `__getitem()__` should provide
                single example/label pairs as a dictionary under the keys
                'original' and 'target'.
            val_set: Optional[In1FlexDataset]
                The optional data set to validate on. Output is written to the
                directory named val_imgs relative to the current working
                directory. No validation is done when given `None` (the default
                value).
            init_epoch: int (Default 1)
                What iteration the training should start at. Epochs start at 1, not 0, to
                be more human-readable.
        """
        raise NotImplementedError("Abstract method")
