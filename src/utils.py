import random
from hydra.utils import instantiate
import torch
import torch.nn.functional as F
from src.models import Ensemble
from src.similarity_functions import Similarity

device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


def s3e_ensemble(cfg):
    similarity_measure = Similarity(cfg)
    ensemble = Ensemble(cfg, similarity_measure, device)

    # Batched generation of uncertanties
    h_s3e = ensemble.get_h_s3e_y_batched()
    h_s3e_theta = ensemble.get_h_s3e_y_theta_batched()

    # There's also a naive implementation (takes very long!)
    # h_s3e = ensemble.get_h_s3e_y()
    # h_s3e_theta = ensemble.get_h_s3e_y_theta()

    print(h_s3e, h_s3e_theta)
    print(h_s3e - h_s3e_theta)
