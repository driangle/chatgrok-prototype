from os import path
import logging

DEFAULT_PROMPT_EXPERIMENT = "1"


class PromptLoader:

    def __init__(self, prompt_experiment):
        self._prompt_experiment = prompt_experiment if prompt_experiment else DEFAULT_PROMPT_EXPERIMENT
        logging.info(f"Running Prompt Experiment [{self._prompt_experiment}]")

    def load(self, prompt_name, required=True):
        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(
            basepath,
            f'./prompts/{self._prompt_experiment}/{prompt_name}.txt'
        ))
        logging.debug(f"Loading prompt from [{filepath}]")
        try:
            with open(filepath, 'r') as prompt_file:
                return prompt_file.read().strip()
        except FileNotFoundError as e:
            if required:
                raise e
            return None
