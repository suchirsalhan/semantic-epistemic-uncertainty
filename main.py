from collections import defaultdict
import wandb
import torch
import hydra
import random
import numpy as np
from trl import setup_chat_format
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from hydra.utils import instantiate
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import logging

from src.generations import analyse_generations, collect_generations


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default"
)
def main(cfg: DictConfig):
    load_dotenv()

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg.dataset)

    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.wandb_mode  # NOTE: disabled by default
    )

    ensemble = []
    # instantiate all selected models and
    # their tokenizers for the ensemble
    for name, config in cfg.models.items():
        model = instantiate(config.model_spec)
        tokenizer = instantiate(config.tokenizer_spec)
        ensemble.append((model, tokenizer))

    dataset = instantiate(cfg.dataset.spec)
    generations = collect_generations(ensemble, dataset, cfg)
    individual_entropies = analyse_generations(generations)


if __name__ == '__main__':
    main()
