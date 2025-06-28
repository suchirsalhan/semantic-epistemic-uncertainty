from random import choice, choices
from hydra.utils import instantiate
import torch.nn.functional as F
from tqdm import tqdm
import random
import torch
from pprint import pprint


class Ensemble:
    def __init__(self, cfg, similarity_measure, device):
        self.cfg = cfg
        self.device = device
        self.similarity_measure = similarity_measure
        self.ensemble = []

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

    def sample(self, entry, num_repeats: int, prompt: str):
        """
            Generating a single sample from p(y | theta) distribution
        """

        model, tokenizer = entry
        # ^
        # theta

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generated = []

        with torch.no_grad():
            # batching the generation process
            # because the naive and batched implementations
            # scale the number of genrations as O(n^2) and O(n^3)
            # for computing the uncertainties, where n is the
            # number of monte carlo samples
            # so e.g. if n=30, n^3=2700 which cannot
            # easily fit into a GPU memory
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

    def get_h_s3e_y_batched(self, prompt='What\'s the weather like tomorrow?'):
        """
        Batched generation sequential implementation of H_S3E(Y)

        H_{S3E}(Y) =−E_{y~p(y)} log E_{y'~p(y)} S(y,y') 
                     ^              ^           ^
                outer model    inner model   similarity function
        """

        # outer models are used to estimate
        # the expected value E_{y~p(y)}
        # i.e. to sample y
        outer_models = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo
        )

        # inner models are used to estimate
        # the expected value E_{y'~p(y)}
        # i.e. to sample y' (y_prime)
        inner_models = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo ** 2
        )
        model_generations = {}
        # storing all generations to be able
        # to precompute similarity metrics
        all_generations = []

        # first count how many times a model is chosen
        # from the ensemble to generate either y or y' (y_prime)
        # i.e. how many times the model is used to estimate
        # either E_{y~p(y)} or E_{y'~p(y)}
        # Next batch generate outputs
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
            # iterable object will allow us to 'take'
            # a model generation without having to account
            # for any already used generations
            model_generations[model] = iter(model_generations[model])

        # precomputing similarities between all possible y and y' (y_prime) values
        similarity_matrix, indices = self.similarity_measure.encode_generated(
            all_generations
        )

        # same as for model_generations, an iterable
        # object will allow us to select a model
        # for sampling y~p(y) and y'~p(y) (y_prime)
        # without having to manually account for
        # the order of the selections
        outer_models = iter(outer_models)
        inner_models = iter(inner_models)

        # accumulator for estimating E_{y~p(y)}
        s_outer = 0

        # estimate expectation over num_monte_carlo samples
        for _ in range(self.cfg.generation.num_monte_carlo):
            # selecting a model for sampling y~p(y)
            outer_model = next(outer_models)
            # selecting a pre-generated sample y~p(y)
            y = next(model_generations[outer_model])
            # finding the row number/column number of y in similarity_matrix
            y_idx = indices[y]
            # accumulator for estimating E_{y'~p(y)} (y_prime)
            s_inner = 0
            # estimate expectation over num_monte_carlo samples
            for _ in range(self.cfg.generation.num_monte_carlo):
                # selecting a model for sampling y'~p(y) (y_prime)
                inner_model = next(inner_models)
                # selecting a pre-generated sample y'~p(y) (y_prime)
                y_prime = next(model_generations[inner_model])
                # finding the row number/column number
                # of y' (y_prime) in similarity_matrix
                y_prime_idx = indices[y_prime]
                # adding a precomputed similarity of y and y' (y_prime)
                s_inner += similarity_matrix[y_idx, y_prime_idx]
            s_inner /= self.cfg.generation.num_monte_carlo
            s_inner = torch.log(s_inner)
            s_outer += s_inner
        s_outer /= self.cfg.generation.num_monte_carlo
        s_outer *= -1
        return s_outer

    def get_h_s3e_y_theta_batched(self, prompt='What\'s the weather like tomorrow?'):
        """
        Batched generation sequential implementation of H_S3E(Y|\Theta)

        H_{S3E}(Y|\Theta) =−E{theta~\Theta} E{y~p(y|theta)} log E{y'~p(y|theta)} S(y,y')
                             ^
                     candidate_models
        """
        # candidate_models are used in computing
        # the leftmost expectation in H_{S3E}(Y|\Theta)
        candidate_models = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo
        )

        model_generations = {}
        # storing all generations to be able
        # to precompute similarity metrics
        all_generations = []

        # first count how many times a model is chosen
        # from the ensemble to generate an output
        # next batch generate outputs
        for model in tqdm(set(candidate_models)):
            count = candidate_models.count(model)
            # for a model theta, we need to generate
            # m * (n^2 + n)
            # where m is how many times it is selected
            #   to be in candidate_models, i.e. how many
            #   this particular model is used in estimating
            #   E{theta~\Theta}
            # where n is the number of monte carlo samples
            #   used to estimate the other two expectations
            #   i.e. E{y~p(y|theta)} and E{y'~p(y|theta)}
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
            # iterable object will allow us to 'take'
            # a model generation without having to account
            # for any already used generations
            model_generations[model] = iter(model_generations[model])

        # same as for model_generations, an iterable
        # object will allow us to select a model
        # for sampling theta~\Theta
        # without having to manually account for
        # the order of the selections
        candidate_models = iter(candidate_models)
        # precomputing similarities between all possible y and y' (y_prime) values
        similarity_matrix, indices = self.similarity_measure.encode_generated(
            all_generations
        )

        # accumulator for estimating E_{theta~\Theta}
        s = 0
        # estimate expectation over num_monte_carlo samples
        for _ in range(self.cfg.generation.num_monte_carlo):
            # select theta~\Theta
            model = next(candidate_models)
            # accumulator for estimating E_{y~p(y|theta)}
            s_outer = 0
            for _ in range(self.cfg.generation.num_monte_carlo):
                # select precomputed y~p(y | theta)
                y = next(model_generations[model])
                # finding the row number/column number
                # of y in similarity_matrix
                y_idx = indices[y]
                # accumulator for estimating E_{y~p(y'|theta)} (y_prime)
                s_inner = 0
                for _ in range(self.cfg.generation.num_monte_carlo):
                    y_prime = next(model_generations[model])
                    # finding the row number/column number
                    # of y' (y_prime) in similarity_matrix
                    y_prime_idx = indices[y_prime]
                    # adding a precomputed similarity of y and y' (y_prime)
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
        """
            Function for estimating the probability P(y | theta)
        """
        # tokenize the input
        tokenized_y = tokenizer(y, return_tensors='pt').to(self.device)
        # compute model logits for each sequence of tokens
        # [
        #   P(y_0 | prompt), P(y_1 | prompt, y_0),
        #   ...,
        #   P(y_n | prompt, y_0, y_1, ..., y_{n-1})
        # ]
        # given the actual sequence tokens as labels
        logits = theta(**tokenized_y, labels=tokenized_y['input_ids']).logits

        # code based on https://github.com/huggingface/transformers/blob/f1d822ba337499d429f832855622b97d90ac1406/src/transformers/models/llama/modeling_llama.py#L1205-L1210
        # TODO: need help here
        shift_logits = F.log_softmax(
            logits[..., :-1, :].contiguous().squeeze(0), dim=1
        )
        shift_labels = tokenized_y['input_ids'][..., 1:].contiguous()
        probability = 0
        for idx, label in enumerate(shift_labels[0]):
            probability += shift_logits[idx, label]
        probability /= len(tokenized_y['input_ids'][0])
        return torch.exp(probability)

    # P(y) ? E_{theta ~ \Theta} P(y | theta)

    def batch_compute_entropies(self, prompt='What\'s the weather like tomorrow?'):
        """
            Function for generating a batch of samples from 
            the p(y) distribution and estimating 
            h_s3e_y and h_s3e_y_theta from the generations
        """

        mc_y_samples = []

        # choose models to generate y-s from
        mc_y_ensemble_candidates = choices(
            self.ensemble, k=self.cfg.generation.num_monte_carlo
        )

        # generate as many y-s per ensemble model
        # as many times it has been chosen for the
        # mc_y_ensemble_candidates list
        for entry in set(mc_y_ensemble_candidates):
            mc_y_samples.extend(
                self.sample(
                    entry,
                    mc_y_ensemble_candidates.count(entry),
                    prompt
                )
            )

        # shuffle because otherwise the y samples would
        # be generated in order of the models occuring
        # in set(mc_y_ensemble_candidates)
        random.Random(self.cfg.seed).shuffle(mc_y_samples)

        similarity_matrix_raw, indices = self.similarity_measure.encode_generated(
            mc_y_samples
        )

        h_s3e = similarity_matrix_raw.mean(dim=0).log().mean() * -1

        # set S(y,y) to zero -- will implicity be skipping
        # the summation over indices i=j
        similarity_matrix = similarity_matrix_raw.fill_diagonal_(0.0).cpu()

        p_ensemble = {}
        p_model = {
            model: {}
            for model in set(mc_y_ensemble_candidates)
        }

        # precompute p(y) and p(y | theta)
        # since y-s can occur multiple times in
        # mc_y_samples, no need to recompute its
        # probabilities -- therefore iterating
        # over set(mc_y_samples) instead of
        # mc_y_samples is fine
        for y in tqdm(
            set(mc_y_samples),
            desc="Computing probabilities p(y) and p(y | theta)"
        ):
            accumulator = 0
            for model in p_model:
                p_y_theta = self.p_y_given_theta(
                    y, model[0], model[1]
                )
                accumulator += p_y_theta
                p_model[model][y] = p_y_theta
            p_ensemble[y] = accumulator / len(p_model)

        h_s3e_theta = 0
        for model in p_model:
            p_model_vals = torch.tensor(
                list(p_model[model].values())
            )
            p_ensemble_vals = torch.tensor(
                list(p_ensemble.values())
            )
            Z = p_model_vals / p_ensemble_vals

            # construct a matrix that is Z everywhere
            # and 0 on the diagonal -- this allows
            # us to easily get Z_j by summing over
            # the entire row, without worrying
            # about indices i and j
            Z_j_s = Z.repeat(
                Z.shape[0], 1
            ).fill_diagonal_(0.0).sum(dim=1)
            Z = Z.sum()

            assert Z_j_s.shape == p_ensemble_vals.shape == p_model_vals.shape

            inner_importance_weights = p_model_vals / (p_ensemble_vals * Z_j_s)
            #         ^
            #   p(y_i | theta)
            #   --------------
            #    p(y_i) * Z_j

            model_similarity_matrix = similarity_matrix * inner_importance_weights

            inner_sum = model_similarity_matrix.sum(
                dim=0) / (len(p_ensemble) - 1)
            log_arg = (p_model_vals * inner_sum) / (p_ensemble_vals * Z)
            # NOTE: dividing by 10 here makes
            # the result (somewhat?) sensible
            logarithm = torch.log(log_arg)
            outer_sum = logarithm.mean()
            h_s3e_theta += outer_sum

        h_s3e_theta /= len(p_model)
        h_s3e_theta *= -1
        return h_s3e, h_s3e_theta

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
