"""The lightly_plus_time.lightly.models package provides model implementations.

Note that the high-level building blocks will be deprecated with 
lightly version 1.3.0. Instead, use low-level building blocks to build the
models yourself.

Example implementations for all models can be found here:
`Model Examples <https://docs.lightly_plus_time.lightly.ai/examples/models.html>`_

The package contains an implementation of the commonly used ResNet and
adaptations of the architecture which make self-supervised learning simpler.

The package also hosts the Lightly model zoo - a list of downloadable ResNet
checkpoints.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly_plus_time.lightly.models.resnet import ResNetGenerator
from lightly_plus_time.lightly.models.barlowtwins import BarlowTwins
from lightly_plus_time.lightly.models.simclr import SimCLR
from lightly_plus_time.lightly.models.simsiam import SimSiam
from lightly_plus_time.lightly.models.byol import BYOL
from lightly_plus_time.lightly.models.moco import MoCo
from lightly_plus_time.lightly.models.nnclr import NNCLR
from lightly_plus_time.lightly.models.zoo import ZOO
from lightly_plus_time.lightly.models.zoo import checkpoints

from lightly_plus_time.lightly.models import utils