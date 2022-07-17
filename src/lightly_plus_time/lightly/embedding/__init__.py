"""The lightly_plus_time.lightly.embedding module provides trainable embedding strategies.

The embedding models use a pre-trained ResNet but should be finetuned on each
dataset instance.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

#Updated for TS project- GA
# from lightly_plus_time.lightly.embedding._base import BaseEmbedding
# from lightly_plus_time.lightly.embedding.embedding import SelfSupervisedEmbedding
from lightly_plus_time.lightly.embedding._base import BaseEmbedding
from lightly_plus_time.lightly.embedding.embedding import SelfSupervisedEmbedding