from src.utils import format_dataset_to_prompts, get_prompt


def collect_generations(ensemble, dataset, cfg):
    prompt_dataset, inference_dataset = format_dataset_to_prompts(
        dataset, cfg.generation, cfg.dataset
    )

    print(get_prompt(prompt_dataset))

    # for model, tokenizer in ensemble:
    #     pass


def analyse_generations(generations):
    pass
