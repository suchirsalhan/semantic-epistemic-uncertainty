from collections import defaultdict
import pickle
import wandb
import torch
import hydra
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from hydra.utils import instantiate
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import logging


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

    instantiate(
        cfg.uncertainty.eval_function, cfg
    )


if __name__ == '__main__':
    main()
