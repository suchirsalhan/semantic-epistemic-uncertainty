from random import choice, choices
from hydra.utils import instantiate
import torch.nn.functional as F
from tqdm import tqdm
import random
import torch


class Ensemble:
    def __init__(self, cfg, similarity_measure, device):
        self.cfg = cfg
        self.device = device
        self.similarity_measure = similarity_measure
        self.ensemble = []
        # store intermediate results from each
        # member of the ensemble
        self.ensemble_generations = {}
        self.ensemble_entropies = {}
        self.ensemble_analysis = {}

        for name, config in cfg.models.items():
            model = instantiate(config.model_spec).to(device)
            tokenizer = instantiate(config.tokenizer_spec)
            self.ensemble.append((model, tokenizer))

    def __call__(self, prompt, ensemble_entry=None):
        """
            Generating a single sample from p(y) distribution
        """
        if ensemble_entry is None:
            ensemble_entry = choice(self.ensemble)
        theta, tokenizer = ensemble_entry
        model_inputs = tokenizer(
            [prompt],
            return_tensors="pt"
        ).to(self.device)
        y = theta.generate(
            **model_inputs, do_sample=True, num_beams=1
        )
        return tokenizer.batch_decode(y, skip_special_tokens=True)[0]

    def get_h_s3e_y_naive(self, prompt='What\'s the weather like tomorrow?'):
        """Naive sequential implementation of H_S3E(Y)"""
        s_outer = 0
        for _ in range(self.cfg.generation.num_monte_carlo):
            y = self(prompt)
            s_inner = 0
            for _ in range(self.cfg.generation.num_monte_carlo):
                y_prime = self(prompt)
                # computing similarity of y and y'
                s_inner += self.similarity_measure(y, y_prime)
            s_inner /= self.cfg.generation.num_monte_carlo
            s_inner = torch.log(s_inner)
            s_outer += s_inner
        s_outer /= self.cfg.generation.num_monte_carlo
        s_outer *= -1
        return s_outer

    def get_h_s3e_y_batched(self, prompt='What\'s the weather like tomorrow?'):
        """Batched generation sequential implementation of H_S3E(Y)"""
        outer_models = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo
        )
        inner_models = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo ** 2
        )
        model_generations = {}
        all_generations = []

        # first count how many times a model is chosen
        # from the ensemble to generate an output
        # next batch generate outputs
        for model in tqdm(set(outer_models) | set(inner_models)):
            count = outer_models.count(
                model
            ) + inner_models.count(model)

            model_generations[model] = self.sample(
                # args:
                # key: (model, tokenizer) tuple
                # count: number of times key is invoked
                # prompt: is just a prompt
                model, count, prompt
            )
            all_generations.extend(model_generations[model])
            model_generations[model] = iter(model_generations[model])

        similarity_matrix, indices = self.similarity_measure.encode_generated(
            all_generations
        )

        outer_models = iter(outer_models)
        inner_models = iter(inner_models)

        s_outer = 0
        for _ in range(self.cfg.generation.num_monte_carlo):
            outer_model = next(outer_models)
            y = next(model_generations[outer_model])
            y_idx = indices[y]
            s_inner = 0
            for _ in range(self.cfg.generation.num_monte_carlo):
                inner_model = next(inner_models)
                y_prime = next(model_generations[inner_model])
                y_prime_idx = indices[y_prime]
                # computing similarity of y and y'
                s_inner += similarity_matrix[y_idx, y_prime_idx]
            s_inner /= self.cfg.generation.num_monte_carlo
            s_inner = torch.log(s_inner)
            s_outer += s_inner
        s_outer /= self.cfg.generation.num_monte_carlo
        s_outer *= -1
        return s_outer

    def get_h_s3e_y_theta_naive(self, prompt='What\'s the weather like tomorrow?'):
        """Naive sequential implementation of H_S3E(Y|\Theta)"""
        s = 0
        for _ in range(self.cfg.generation.num_monte_carlo):
            ensemble_entry = choice(self.ensemble)
            s_outer = 0
            for _ in range(self.cfg.generation.num_monte_carlo):
                y = self(prompt, ensemble_entry)
                s_inner = 0
                for _ in range(self.cfg.generation.num_monte_carlo):
                    y_prime = self(prompt, ensemble_entry)
                    s_inner += self.similarity_measure(y, y_prime)
                s_inner /= self.cfg.generation.num_monte_carlo
                s_inner = torch.log(s_inner)
                s_outer += s_inner
            s_outer /= self.cfg.generation.num_monte_carlo
            s += s_outer
        s /= self.cfg.generation.num_monte_carlo
        s *= -1
        return s

    def get_h_s3e_y_theta_batched(self, prompt='What\'s the weather like tomorrow?'):
        """Batched generation sequential implementation of H_S3E(Y|\Theta)"""
        candidate_models = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo
        )
        model_generations = {}
        all_generations = []

        # first count how many times a model is chosen
        # from the ensemble to generate an output
        # next batch generate outputs
        for model in tqdm(set(candidate_models)):
            count = candidate_models.count(model)
            model_generations[model] = self.sample(
                # args:
                # model: (model, tokenizer) tuple
                # model_generations[key]: number of times key is invoked
                # prompt: is just a prompt
                model,
                count * (
                    self.cfg.generation.num_monte_carlo ** 2 +
                    self.cfg.generation.num_monte_carlo
                ),
                prompt
            )
            all_generations.extend(model_generations[model])
            model_generations[model] = iter(model_generations[model])

        candidate_models = iter(candidate_models)
        similarity_matrix, indices = self.similarity_measure.encode_generated(
            all_generations
        )

        s = 0
        for _ in range(self.cfg.generation.num_monte_carlo):
            model = next(candidate_models)
            s_outer = 0
            for _ in range(self.cfg.generation.num_monte_carlo):
                y = next(model_generations[model])
                y_idx = indices[y]
                s_inner = 0
                for _ in range(self.cfg.generation.num_monte_carlo):
                    y_prime = next(model_generations[model])
                    y_prime_idx = indices[y_prime]
                    s_inner += similarity_matrix[y_idx, y_prime_idx]
                s_inner /= self.cfg.generation.num_monte_carlo
                s_inner = torch.log(s_inner)
                s_outer += s_inner
            s_outer /= self.cfg.generation.num_monte_carlo
            s += s_outer
        s /= self.cfg.generation.num_monte_carlo
        s *= -1
        return s.item()

    def p_y_given_theta(self, y, theta, tokenizer):
        tokenized_y = tokenizer(y, return_tensors='pt').to(self.device)
        logits = theta(**tokenized_y, labels=tokenized_y['input_ids']).logits

        # code based on https://github.com/huggingface/transformers/blob/f1d822ba337499d429f832855622b97d90ac1406/src/transformers/models/llama/modeling_llama.py#L1205-L1210
        shift_logits = F.log_softmax(
            logits[..., :-1, :].contiguous().squeeze(0), dim=1
        )
        shift_labels = tokenized_y['input_ids'][..., 1:].contiguous()
        probability = 0
        for idx, label in enumerate(shift_labels[0]):
            probability += shift_logits[idx, label]
        return probability.item()

    def batch_compute_entropies(self, prompt='What\'s the weather like tomorrow?'):
        """
            Function for generating a batch of samples from 
            the p(y) distribution and estimating 
            h_s3e_y and h_s3e_y_theta from the generations
        """

        # generating more samples than needed to reduce the risk
        # of using the same sample too many times
        # (TODO: maths to estimate a good number of generations)

        mc_y_ensemble_candidates = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo
        )
        mc_y_samples = []
        for entry in set(mc_y_ensemble_candidates):
            mc_y_samples.extend(
                self.sample(
                    entry, mc_y_ensemble_candidates.count(entry)
                )
            )

        random.Random(self.cfg.seed).shuffle(mc_y_samples)

        # computing h_s3e_y
        # TODO: vectorise
        s_outer = 0
        for _ in range(self.cfg.generation.num_monte_carlo):
            y = choice(mc_y_samples)
            s_inner = 0
            for _ in range(self.cfg.generation.num_monte_carlo):
                y_prime = choice(mc_y_samples)
                # computing similarity of y and y'
                s_inner += self.similarity_measure(y, y_prime)
            s_inner /= self.cfg.generation.num_monte_carlo
            s_inner = torch.log(s_inner)
            s_outer += s_inner
        s_outer /= self.cfg.generation.num_monte_carlo
        s_outer *= -1
        h_s3e_y = s_outer.item()

        # computing h_s3e_y_theta
        # TODO: vectorise
        for entry in self.ensemble:
            model, tokenizer = entry
            denominator_models = choices(self.ensemble)
            Z_numerator = torch.tensor(
                list(
                    map(
                        lambda x: self.p_y_given_theta(
                            x, model, tokenizer
                        ), mc_y_samples
                    )
                )
            )
            Z_denominator = torch.tensor(
                list(
                    map(
                        lambda x: self.p_y_given_theta(
                            x[1],
                            denominator_models[x[0]][0],
                            denominator_models[x[0]][1]
                        ),
                        enumerate(y_is)
                    )
                )
            )
            Z = Z_numerator / Z_denominator
            for i in range(self.cfg.generation.num_monte_carlo):
                for j in range(self.cfg.generation.num_monte_carlo):
                    if i == j:
                        continue

        s = 0
        for _ in range(self.cfg.generation.num_monte_carlo):
            ensemble_entry = choice(self.ensemble)
            s_outer = 0
            for _ in range(self.cfg.generation.num_monte_carlo):
                y = self(prompt, ensemble_entry)
                s_inner = 0
                for _ in range(self.cfg.generation.num_monte_carlo):
                    y_prime = self(prompt, ensemble_entry)
                    s_inner += self.similarity_measure(y, y_prime)
                s_inner /= self.cfg.generation.num_monte_carlo
                s_inner = torch.log(s_inner)
                s_outer += s_inner
            s_outer /= self.cfg.generation.num_monte_carlo
            s += s_outer
        s /= self.cfg.generation.num_monte_carlo
        s *= -1

        samples = list(map(self.sample, mc_y_samples))
        return samples

    def sample(self, entry, num_repeats: int, prompt: str):
        model, tokenizer = entry

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generated = []

        with torch.no_grad():
            for idx in tqdm(range(0, num_repeats, self.cfg.generation.batch_size)):
                num_samples = min(
                    self.cfg.generation.batch_size,
                    num_repeats - idx
                )
                model_inputs = tokenizer(
                    [prompt] * num_samples,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                generated_ids = model.generate(
                    **model_inputs, do_sample=True, num_beams=1
                )
                batch = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                generated.extend(batch)

        return generated
