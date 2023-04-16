import os
import logging
from .worker import GPTWorker
from .grounding import GPTGrounding
from .prompt_loader import PromptLoader

logger = logging.getLogger('chatgrok.multilane')

class Multilane:

    def __init__(self, doc_chunks, prompt_experiment):
        prompt_loader = PromptLoader(prompt_experiment)
        self._workers = [
            GPTWorker(os.environ['GPT_MODEL'], doc_chunk, prompt_loader)
            for doc_chunk in doc_chunks
        ]
        self._grounding = GPTGrounding(os.environ['GPT_MODEL'], prompt_loader)

    def call(self, query):
        worker_outputs = [worker.ask(query) for worker in self._workers]
        logger.debug(f"Grounding...")
        return self._grounding.ask(query, worker_outputs)
