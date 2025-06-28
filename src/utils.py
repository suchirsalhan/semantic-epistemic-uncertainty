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

    h_s3e, h_s3e_theta = ensemble.batch_compute_entropies()

    # Batched generation of uncertanties
    print(h_s3e, h_s3e_theta)

    print('--------')
    print('now batch')

    h_s3e = ensemble.get_h_s3e_y_batched()
    h_s3e_theta = ensemble.get_h_s3e_y_theta_batched()
    print(h_s3e, h_s3e_theta)
    # print(h_s3e - h_s3e_theta)
