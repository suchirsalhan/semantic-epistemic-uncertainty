from collections import defaultdict
import functools
import pickle
from src.semantic_uncertainty.uncertainty.utils.eval_utils import accuracy_at_quantile, area_under_thresholded_accuracy, auroc, bootstrap, compatible_bootstrap
from src.semantic_uncertainty.analyze_results import analyze_run
from src.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta, EntailmentGPT35, EntailmentGPT4, EntailmentGPT4Turbo, EntailmentLlama, cluster_assignment_entropy, context_entails_response, get_semantic_ids, logsumexp_by_id, predictive_entropy, predictive_entropy_rao
from src.semantic_uncertainty.uncertainty.data.data_utils import load_ds
from src.semantic_uncertainty.uncertainty.utils import utils
from src.utils import format_dataset_to_prompts, get_prompt
from tqdm import tqdm
import numpy as np
import logging
import random
import torch
import os
import gc


def collect_generations(model_cfg, cfg):
    # Setup run.
    if cfg.dataset.name == 'svamp':
        if not cfg.generation.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            cfg.generation.use_context = True
    elif cfg.dataset.name == 'squad':
        if not cfg.dataset.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            cfg.generation.answerable_only = True

    experiment_details = {'args': cfg}
    random.seed(cfg.seed)
    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    # Get accuracy metric.
    metric = utils.get_metric(cfg.generation.metric)

    """NOTE: LOADING DATASET"""

    # Load dataset.
    train_dataset, validation_dataset = load_ds(
        cfg.dataset.name, seed=cfg.seed
    )
    # TODO: check out ood train dataset
    # if cfg.dataset.ood_train_dataset is not None:
    #     logging.warning(
    #         'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
    #         cfg.dataset.ood_train_dataset)
    #     # Get indices of answerable and unanswerable questions and construct prompt.
    #     train_dataset, _ = load_ds(
    #         args.ood_train_dataset, add_options=args.use_mc_options)
    # if not isinstance(train_dataset, list):
    #     logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(
        train_dataset
    )

    if cfg.generation.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(
            validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(
        answerable_indices, cfg.generation.num_few_shot
    )
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(cfg.generation)
    BRIEF = utils.BRIEF_PROMPTS[cfg.generation.brief_prompt]
    arg = cfg.generation.brief_response  # if cfg.generation.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    model = utils.init_model(model_cfg)

    # Initialize prompt for p_true baseline.
    # if cfg.generation.compute_p_true:
    #     logging.info(80*'#')
    #     logging.info('Constructing few-shot prompt for p_true.')

    #     p_true_indices = random.sample(
    #         answerable_indices, cfg.generation.p_true_num_fewshot)
    #     remaining_answerable = list(
    #         set(remaining_answerable) - set(p_true_indices))
    #     p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
    #         model=model, dataset=train_dataset, indices=p_true_indices,
    #         prompt=prompt, brief=BRIEF,
    #         brief_always=args.brief_always and args.enable_brief,
    #         make_prompt=make_prompt, num_generations=args.num_generations,
    #         metric=metric)
    #     wandb.config.update(
    #         {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
    #     wandb.log(dict(len_p_true=len_p_true))
    #     experiment_details['p_true_indices'] = p_true_indices
    #     experiment_details['p_true_responses'] = p_true_responses
    #     experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
    #     logging.info('Finished constructing few-shot prompt for p_true.')
    #     logging.info(80*'#')
    #     logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
    #     logging.info(80*'#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    split_generations = {}
    split_results = {}
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            if not cfg.generation.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(
                set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(
            cfg.generation.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if cfg.generation.num_samples > len(dataset):
            logging.warning(
                'Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.mps.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {
                'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, BRIEF, cfg.generation.brief_response)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            if dataset_split == 'train' and cfg.generation.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = cfg.generation.num_generations + 1

            for i in range(num_generations):

                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else cfg.generation.temperature

                predicted_answer, token_log_likelihoods, embedding = model.predict(
                    local_prompt, temperature)
                embedding = embedding.cpu() if embedding is not None else None

                # Only compute accuracy if question is answerable.
                compute_acc = cfg.generation.compute_accuracy_at_all_temps or (
                    i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0  # pylint: disable=invalid-name

                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    if cfg.generation.use_context:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info(
                        'low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(
                        15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example)})

                else:
                    logging.info('high-t prediction '.ljust(15) +
                                 str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

            if cfg.generation.compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')
        split_generations[dataset_split] = generations

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        # wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            if cfg.generation.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            # utils.save(results_dict, 'uncertainty_measures.pkl')
        split_results[dataset_split] = results_dict

    # utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model

    return split_results, split_generations


def compute_uncertainty_measures_for_generations(split_result_dict, split_generations, cfg):
    # Load entailment model.
    if cfg.uncertainty.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if cfg.uncertainty.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif cfg.uncertainty.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(
                cfg.uncertainty.entailment_cache_id,
                cfg.uncertainty.entailment_cache_only
            )
        elif cfg.uncertainty.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(
                cfg.uncertainty.entailment_cache_id,
                cfg.uncertainty.entailment_cache_only
            )
        elif cfg.uncertainty.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(
                cfg.uncertainty.entailment_cache_id,
                cfg.uncertainty.entailment_cache_only
            )
        elif 'llama' in cfg.uncertainty.entailment_model.lower():
            entailment_model = EntailmentLlama(
                cfg.uncertainty.entailment_cache_id,
                cfg.uncertainty.entailment_cache_only,
                cfg.uncertainty.entailment_model
            )
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    if cfg.uncertainty.compute_p_true_in_compute_stage:
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        del p_true_responses
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    if cfg.uncertainty.recompute_accuracy:
        # This is usually not enabled.
        logging.warning(
            'Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(cfg.uncertainty.metric)

    train_generations = split_generations['train']

    # Restore outputs from `generate_answrs.py` run.
    result_dict = split_result_dict['validation']
    result_dict['semantic_ids'] = []

    validation_generations = split_generations['validation']

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not cfg.uncertainty.use_all_generations:
            if cfg.uncertainty.use_num_generations == -1:
                raise ValueError
            responses = [fr[0]
                         for fr in full_responses[:cfg.uncertainty.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if cfg.uncertainty.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if cfg.uncertainty.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not cfg.uncertainty.use_all_generations:
                log_liks = [r[1]
                            for r in full_responses[:cfg.uncertainty.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if cfg.uncertainty.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if cfg.uncertainty.condition_on_question and cfg.uncertainty.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=cfg.uncertainty.strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(
                cluster_assignment_entropy(semantic_ids))

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            entropies['regular_entropy'].append(
                predictive_entropy(log_liks_agg))

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(
                semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join(
                [f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if cfg.uncertainty.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= cfg.uncertainty.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    logging.info('Raw accuracies on original task: %f', validation_is_true)

    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f',
                 np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if cfg.uncertainty.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if cfg.uncertainty.compute_p_ik or cfg.uncertainty.compute_p_ik_answerable:
        # Assemble training data for embedding classification.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [
            0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f',
                     np.mean(train_unanswerable))

    if cfg.uncertainty.compute_p_ik:
        logging.info('Starting training p_ik on train embeddings.')
        # Train classifier of correct/incorrect from embeddings.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        logging.info('Finished training p_ik on train embeddings.')

    if cfg.uncertainty.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if cfg.uncertainty.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [
            1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [
            1 - np.exp(p) for p in p_trues]

    if cfg.uncertainty.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    return result_dict


def analyse_generations(results_old, cfg):
    # Set up evaluation metrics.
    if cfg.analysis.answer_fractions_mode == 'default':
        answer_fractions = [0.8, 0.9, 0.95, 1.0]
    elif cfg.analysis.answer_fractions_mode == 'finegrained':
        answer_fractions = [round(i, 3) for i in np.linspace(0, 1, 20+1)]
    else:
        raise ValueError

    rng = np.random.default_rng(cfg.seed)
    eval_metrics = dict(zip(
        ['AUROC', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
        list(zip(
            [auroc, area_under_thresholded_accuracy, np.mean],
            [compatible_bootstrap, compatible_bootstrap, bootstrap]
        )),
    ))
    for answer_fraction in answer_fractions:
        key = f'accuracy_at_{answer_fraction}_answer_fraction'
        eval_metrics[key] = [
            functools.partial(accuracy_at_quantile, quantile=answer_fraction),
            compatible_bootstrap]

    result_dict = {'performance': {}, 'uncertainty': {}}

    # First: Compute simple accuracy metrics for model predictions.
    all_accuracies = dict()
    all_accuracies['accuracy'] = 1 - \
        np.array(results_old['validation_is_false'])

    bootstrap_fn = bootstrap(np.mean, rng)
    for name, target in all_accuracies.items():
        # print(name)
        # print(target)
        result_dict['performance'][name] = {}
        result_dict['performance'][name]['mean'] = np.mean(target)
        # result_dict['performance'][name]['bootstrap'] = bootstrap_fn(target)

    rum = results_old['uncertainty_measures']
    if 'p_false' in rum and 'p_false_fixed' not in rum:
        # Restore log probs true: y = 1 - x --> x = 1 - y.
        # Convert to probs --> np.exp(1 - y).
        # Convert to p_false --> 1 - np.exp(1 - y).
        rum['p_false_fixed'] = [1 - np.exp(1 - x) for x in rum['p_false']]

    # Next: Uncertainty Measures.
    # Iterate through the dictionary and compute additional metrics for each measure.
    for measure_name, measure_values in rum.items():
        logging.info('Computing for uncertainty measure `%s`.', measure_name)

        # Validation accuracy.
        validation_is_falses = [
            results_old['validation_is_false'],
            results_old['validation_unanswerable']
        ]

        logging_names = ['', '_UNANSWERABLE']

        # Iterate over predictions of 'falseness' or 'answerability'.
        for validation_is_false, logging_name in zip(validation_is_falses, logging_names):
            name = measure_name + logging_name
            result_dict['uncertainty'][name] = {}

            validation_is_false = np.array(validation_is_false)
            validation_accuracy = 1 - validation_is_false
            if len(measure_values) > len(validation_is_false):
                # This can happen, but only for p_false.
                if 'p_false' not in measure_name:
                    raise ValueError
                logging.warning(
                    'More measure values for %s than in validation_is_false. Len(measure values): %d, Len(validation_is_false): %d',
                    measure_name, len(measure_values), len(validation_is_false))
                measure_values = measure_values[:len(validation_is_false)]

            fargs = {
                'AUROC': [validation_is_false, measure_values],
                'area_under_thresholded_accuracy': [validation_accuracy, measure_values],
                'mean_uncertainty': [measure_values]}

            for answer_fraction in answer_fractions:
                fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [
                    validation_accuracy, measure_values]

            for fname, (function, bs_function) in eval_metrics.items():
                metric_i = function(*fargs[fname])
                result_dict['uncertainty'][name][fname] = {}
                result_dict['uncertainty'][name][fname]['mean'] = metric_i
                logging.info("%s for measure name `%s`: %f",
                             fname, name, metric_i)
                # result_dict['uncertainty'][name][fname]['bootstrap'] = bs_function(
                #     function, rng)(*fargs[fname])

    return result_dict
