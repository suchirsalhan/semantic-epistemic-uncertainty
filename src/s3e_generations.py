from random import choices
import torch

device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


def collect_generations(model_cfg, cfg, ensemble):
    prompt = 'What\'s the weather tomorrow?'

    def sample(entry):
        model, tokenizer = entry
        model_inputs = tokenizer(
            [prompt],
            return_tensors="pt"
        ).to(device)
        generated_ids = model.generate(**model_inputs)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    mc_y_samples = choices(ensemble, k=cfg.generation.num_monte_carlo)
    samples = list(map(sample, mc_y_samples))

    return samples


def compute_uncertainty_measures_for_generations(split_result_dict, split_generations, cfg):
    pass


def analyse_generations(results_old, cfg):
    pass
