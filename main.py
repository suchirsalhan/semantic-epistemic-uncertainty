from collections import defaultdict
import pickle
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
from pprint import pprint
from src.generations import analyse_generations, compute_uncertainty_measures_for_generations, collect_generations


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

    ensemble_generations = {}
    ensemble_entropies = {}
    ensemble_analysis = {}
    # instantiate all selected models and
    # their tokenizers for the ensemble
    for name, config in cfg.models.items():
        # model = instantiate(config.model_spec)
        # tokenizer = instantiate(config.tokenizer_spec)
        split_results, split_generations = collect_generations(config, cfg)
        ensemble_generations[name] = (split_generations, split_results)
        ensemble_entropies[name] = compute_uncertainty_measures_for_generations(
            split_results, split_generations, cfg
        )
        ensemble_analysis[name] = analyse_generations(
            ensemble_entropies[name], cfg)

    with open('all.pickle', 'wb+') as f:
        pickle.dump(
            (
                ensemble_generations,
                ensemble_entropies,
                ensemble_analysis
            ), f
        )

    pprint(ensemble_analysis)


if __name__ == '__main__':
    main()
