import unittest
import torch

import lightly


class TestNestedImports(unittest.TestCase):

    def test_nested_imports(self):
        # active learning
        lightly_plus_time.lightly.active_learning.agents.agent.ActiveLearningAgent
        lightly_plus_time.lightly.active_learning.agents.ActiveLearningAgent
        lightly_plus_time.lightly.active_learning.config.selection_config.SelectionConfig
        lightly_plus_time.lightly.active_learning.config.SelectionConfig
        lightly_plus_time.lightly.active_learning.scorers.classification.ScorerClassification
        lightly_plus_time.lightly.active_learning.scorers.ScorerClassification
        lightly_plus_time.lightly.active_learning.scorers.detection.ScorerObjectDetection
        lightly_plus_time.lightly.active_learning.scorers.ScorerObjectDetection
        lightly_plus_time.lightly.active_learning.utils.bounding_box.BoundingBox
        lightly_plus_time.lightly.active_learning.utils.BoundingBox
        lightly_plus_time.lightly.active_learning.utils.object_detection_output.ObjectDetectionOutput
        lightly_plus_time.lightly.active_learning.utils.ObjectDetectionOutput

        # api imports
        lightly_plus_time.lightly.api.api_workflow_client.ApiWorkflowClient
        lightly_plus_time.lightly.api.ApiWorkflowClient
        lightly_plus_time.lightly.api.bitmask.BitMask

        # data imports
        lightly_plus_time.lightly.data.LightlyDataset
        lightly_plus_time.lightly.data.dataset.LightlyDataset
        lightly_plus_time.lightly.data.BaseCollateFunction
        lightly_plus_time.lightly.data.collate.BaseCollateFunction
        lightly_plus_time.lightly.data.ImageCollateFunction
        lightly_plus_time.lightly.data.collate.ImageCollateFunction
        lightly_plus_time.lightly.data.MoCoCollateFunction
        lightly_plus_time.lightly.data.collate.MoCoCollateFunction
        lightly_plus_time.lightly.data.SimCLRCollateFunction
        lightly_plus_time.lightly.data.collate.SimCLRCollateFunction
        lightly_plus_time.lightly.data.imagenet_normalize
        lightly_plus_time.lightly.data.collate.imagenet_normalize

        # embedding imports
        lightly_plus_time.lightly.embedding.BaseEmbedding
        lightly_plus_time.lightly.embedding._base.BaseEmbedding
        lightly_plus_time.lightly.embedding.SelfSupervisedEmbedding
        lightly_plus_time.lightly.embedding.embedding.SelfSupervisedEmbedding

        # loss imports
        lightly_plus_time.lightly.loss.NTXentLoss
        lightly_plus_time.lightly.loss.ntx_ent_loss.NTXentLoss
        lightly_plus_time.lightly.loss.SymNegCosineSimilarityLoss
        lightly_plus_time.lightly.loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss
        lightly_plus_time.lightly.loss.memory_bank.MemoryBankModule
        lightly_plus_time.lightly.loss.regularizer.CO2Regularizer
        lightly_plus_time.lightly.loss.regularizer.co2.CO2Regularizer

        # models imports
        lightly_plus_time.lightly.models.ResNetGenerator
        lightly_plus_time.lightly.models.resnet.ResNetGenerator
        lightly_plus_time.lightly.models.SimCLR
        lightly_plus_time.lightly.models.simclr.SimCLR
        lightly_plus_time.lightly.models.MoCo
        lightly_plus_time.lightly.models.moco.MoCo
        lightly_plus_time.lightly.models.SimSiam
        lightly_plus_time.lightly.models.simsiam.SimSiam
        lightly_plus_time.lightly.models.ZOO
        lightly_plus_time.lightly.models.zoo.ZOO
        lightly_plus_time.lightly.models.checkpoints
        lightly_plus_time.lightly.models.zoo.checkpoints
        lightly_plus_time.lightly.models.batchnorm.get_norm_layer

        # transforms imports
        lightly_plus_time.lightly.transforms.GaussianBlur
        lightly_plus_time.lightly.transforms.gaussian_blur.GaussianBlur
        lightly_plus_time.lightly.transforms.RandomRotate
        lightly_plus_time.lightly.transforms.rotation.RandomRotate

        # utils imports
        lightly_plus_time.lightly.utils.save_embeddings
        lightly_plus_time.lightly.utils.io.save_embeddings
        lightly_plus_time.lightly.utils.load_embeddings
        lightly_plus_time.lightly.utils.io.load_embeddings
        lightly_plus_time.lightly.utils.load_embeddings_as_dict
        lightly_plus_time.lightly.utils.io.load_embeddings_as_dict
        lightly_plus_time.lightly.utils.fit_pca
        lightly_plus_time.lightly.utils.embeddings_2d.fit_pca

        # core imports
        lightly_plus_time.lightly.train_model_and_embed_images
        lightly_plus_time.lightly.core.train_model_and_embed_images
        lightly_plus_time.lightly.train_embedding_model
        lightly_plus_time.lightly.core.train_embedding_model
        lightly_plus_time.lightly.embed_images
        lightly_plus_time.lightly.core.embed_images