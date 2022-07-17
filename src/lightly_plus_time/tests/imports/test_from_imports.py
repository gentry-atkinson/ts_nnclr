import unittest
import torch

import lightly


class TestFromImports(unittest.TestCase):

    def test_from_imports(self):
        # active learning (commented out don't work)
        from lightly_plus_time.lightly.active_learning.config.selection_config import SelectionConfig
        from lightly_plus_time.lightly.active_learning.agents.agent import ActiveLearningAgent
        from lightly_plus_time.lightly.active_learning.scorers.classification import ScorerClassification

        # api imports
        from lightly_plus_time.lightly.api.api_workflow_client import ApiWorkflowClient
        from lightly_plus_time.lightly.api.bitmask import BitMask

        # data imports
        from lightly_plus_time.lightly.data import LightlyDataset
        from lightly_plus_time.lightly.data.dataset  import LightlyDataset
        from lightly_plus_time.lightly.data  import BaseCollateFunction
        from lightly_plus_time.lightly.data.collate  import BaseCollateFunction
        from lightly_plus_time.lightly.data import ImageCollateFunction
        from lightly_plus_time.lightly.data.collate import ImageCollateFunction
        from lightly_plus_time.lightly.data import MoCoCollateFunction
        from lightly_plus_time.lightly.data.collate import MoCoCollateFunction
        from lightly_plus_time.lightly.data import SimCLRCollateFunction
        from lightly_plus_time.lightly.data.collate import SimCLRCollateFunction
        from lightly_plus_time.lightly.data import imagenet_normalize
        from lightly_plus_time.lightly.data.collate import imagenet_normalize

        # embedding imports
        from lightly_plus_time.lightly.embedding import BaseEmbedding
        from lightly_plus_time.lightly.embedding._base import BaseEmbedding
        from lightly_plus_time.lightly.embedding import SelfSupervisedEmbedding
        from lightly_plus_time.lightly.embedding.embedding import SelfSupervisedEmbedding

        # loss imports
        from lightly_plus_time.lightly.loss import NTXentLoss
        from lightly_plus_time.lightly.loss.ntx_ent_loss import NTXentLoss
        from lightly_plus_time.lightly.loss import SymNegCosineSimilarityLoss
        from lightly_plus_time.lightly.loss.sym_neg_cos_sim_loss import SymNegCosineSimilarityLoss
        from lightly_plus_time.lightly.loss.memory_bank import MemoryBankModule
        from lightly_plus_time.lightly.loss.regularizer import CO2Regularizer
        from lightly_plus_time.lightly.loss.regularizer.co2 import CO2Regularizer

        # models imports
        from lightly_plus_time.lightly.models import ResNetGenerator
        from lightly_plus_time.lightly.models.resnet import ResNetGenerator
        from lightly_plus_time.lightly.models import SimCLR
        from lightly_plus_time.lightly.models.simclr import SimCLR
        from lightly_plus_time.lightly.models import MoCo
        from lightly_plus_time.lightly.models.moco import MoCo
        from lightly_plus_time.lightly.models import SimSiam
        from lightly_plus_time.lightly.models.simsiam import SimSiam
        from lightly_plus_time.lightly.models import ZOO
        from lightly_plus_time.lightly.models.zoo import ZOO
        from lightly_plus_time.lightly.models import checkpoints
        from lightly_plus_time.lightly.models.zoo import checkpoints
        from lightly_plus_time.lightly.models.batchnorm import get_norm_layer

        # transforms imports
        from lightly_plus_time.lightly.transforms import GaussianBlur
        from lightly_plus_time.lightly.transforms.gaussian_blur import GaussianBlur
        from lightly_plus_time.lightly.transforms import RandomRotate
        from lightly_plus_time.lightly.transforms.rotation import RandomRotate

        # utils imports
        from lightly_plus_time.lightly.utils import save_embeddings
        from lightly_plus_time.lightly.utils.io import save_embeddings
        from lightly_plus_time.lightly.utils import load_embeddings
        from lightly_plus_time.lightly.utils.io import load_embeddings
        from lightly_plus_time.lightly.utils import load_embeddings_as_dict
        from lightly_plus_time.lightly.utils.io import load_embeddings_as_dict
        from lightly_plus_time.lightly.utils import fit_pca
        from lightly_plus_time.lightly.utils.embeddings_2d import fit_pca

        # core imports
        from lightly import train_model_and_embed_images
        from lightly import train_embedding_model
        from lightly import embed_images