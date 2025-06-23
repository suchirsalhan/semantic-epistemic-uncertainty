import random
from hydra.utils import instantiate
import torch
import torch.nn.functional as F
from src.similarity_sensitive_entropy.models import Ensemble
from src.similarity_sensitive_entropy.similarity_functions import Similarity

device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


def get_prompt_formatting_fn(cfg, dataset_cfg):
    def formatting_fn(example):
        prompt = ''
        if cfg.brief_response:
            prompt = cfg.brief_prompt
        if cfg.use_context and dataset_cfg.context_key in example:
            prompt += f'Context: {example[dataset_cfg.context_key]}\n'
        prompt += f'Question: {example[dataset_cfg.question_key]}\n'
        prompt += 'Answer:'
        if cfg.append_answer and dataset_cfg.answer_key in example:
            if 'squad' in dataset_cfg.name:
                prompt += f' {example[dataset_cfg.answer_key]["text"][0]}\n\n'
            else:
                prompt += f' {example[dataset_cfg.answer_key]}\n\n'

        return {'prompt': prompt}

    return formatting_fn


def format_dataset_to_prompts(dataset, cfg, dataset_cfg):
    def filter_fn(example):
        if 'svamp' in dataset_cfg.name:
            return len(example[dataset_cfg.answer_key]) > 0
        elif 'squad' in dataset_cfg.name:
            return len(example[dataset_cfg.answer_key]['text']) > 0
        raise NotImplementedError

    dataset = dataset.filter(
        # filter out the empty examples
        filter_fn
    ).shuffle(seed=cfg.seed)

    # the first ${cfg.num_few_shot} examples
    # will be used for the few shot prompt
    prompt_dataset = dataset.select(range(cfg.num_few_shot))

    prompt_format_fn = get_prompt_formatting_fn(cfg, dataset_cfg)
    # format prompt dataset as prompt
    prompt_dataset = prompt_dataset.map(prompt_format_fn)

    # the remaining examples will be used for inference
    inference_dataset = dataset.select(
        range(cfg.num_few_shot, len(dataset))
    )

    return prompt_dataset, inference_dataset


def get_prompt(prompt_dataset):
    return ''.join([entry['prompt'] for entry in prompt_dataset])


def get_h_s3e(cfg, generations):
    h_s3e = 0
    for _ in range(cfg.generation.num_monte_carlo):
        sample_1 = random.choice(generations)
        sample_2 = random.choice(generations)


def oatml_ensemble(cfg):
    # store intermediate results from each
    # member of the ensemble
    ensemble_generations = {}
    ensemble_entropies = {}
    ensemble_analysis = {}
    for name, config in cfg.models.items():
        # generating raw results from an LLM
        # split by ['train', 'test'] dataset splits
        split_results, split_generations = instantiate(
            cfg.uncertainty.collect_generations, config, cfg
        )
        ensemble_generations[name] = (split_generations, split_results)

        # computing semantic entropies for
        # each member of the ensemble
        ensemble_entropies[name] = instantiate(
            cfg.uncertainty.compute_uncertainty_measures_for_generations,
            split_results, split_generations, cfg
        )

        if cfg.uncertainty.analyze_run:
            # analyse the uncertainty results
            ensemble_analysis[name] = instantiate(
                cfg.uncertainty.analyse_generations,
                ensemble_entropies[name], cfg
            )

    return ensemble_generations, ensemble_entropies, ensemble_analysis


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
