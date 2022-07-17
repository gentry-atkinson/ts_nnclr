"""The lightly_plus_time.lightly.models.modules package provides reusable modules.

This package contains reusable modules such as the NNmemoryBankModule which
can be combined with any lightly model.

"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

#Updated for TS project -GA
# from lightly_plus_time.lightly.models.modules.heads import BarlowTwinsProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import BYOLProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import BYOLPredictionHead
# from lightly_plus_time.lightly.models.modules.heads import DINOProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import MoCoProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import NNCLRProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import NNCLRPredictionHead
# from lightly_plus_time.lightly.models.modules.heads import SimCLRProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import SimSiamProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import SimSiamPredictionHead
# from lightly_plus_time.lightly.models.modules.heads import SwaVProjectionHead
# from lightly_plus_time.lightly.models.modules.heads import SwaVPrototypes
# from lightly_plus_time.lightly.models.modules.nn_memory_bank import NNMemoryBankModule

# from lightly import _torchvision_vit_available
# if _torchvision_vit_available:
#     # Requires torchvision >=0.12
#     from lightly_plus_time.lightly.models.modules.masked_autoencoder import MAEBackbone
#     from lightly_plus_time.lightly.models.modules.masked_autoencoder import MAEDecoder
#     from lightly_plus_time.lightly.models.modules.masked_autoencoder import MAEEncoder

from lightly_plus_time.lightly.models.modules.heads import BarlowTwinsProjectionHead
from lightly_plus_time.lightly.models.modules.heads import BYOLProjectionHead
from lightly_plus_time.lightly.models.modules.heads import BYOLPredictionHead
from lightly_plus_time.lightly.models.modules.heads import DINOProjectionHead
from lightly_plus_time.lightly.models.modules.heads import MoCoProjectionHead
from lightly_plus_time.lightly.models.modules.heads import NNCLRProjectionHead
from lightly_plus_time.lightly.models.modules.heads import NNCLRPredictionHead
from lightly_plus_time.lightly.models.modules.heads import SimCLRProjectionHead
from lightly_plus_time.lightly.models.modules.heads import SimSiamProjectionHead
from lightly_plus_time.lightly.models.modules.heads import SimSiamPredictionHead
from lightly_plus_time.lightly.models.modules.heads import SwaVProjectionHead
from lightly_plus_time.lightly.models.modules.heads import SwaVPrototypes
from lightly_plus_time.lightly.models.modules.nn_memory_bank import NNMemoryBankModule

from lightly_plus_time.lightly import _torchvision_vit_available
if _torchvision_vit_available:
    # Requires torchvision >=0.12
    from lightly_plus_time.lightly.models.modules.masked_autoencoder import MAEBackbone
    from lightly_plus_time.lightly.models.modules.masked_autoencoder import MAEDecoder
    from lightly_plus_time.lightly.models.modules.masked_autoencoder import MAEEncoder
