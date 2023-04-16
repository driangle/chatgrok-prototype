import os
import logging
from chatgrok.chains.multilane.worker import GPTWorker
from chatgrok.chains.multilane.grounding import GPTGrounding

class Multilane:

    def __init__(self, doc_chunks):
        self._workers = [
            GPTWorker(os.environ['GPT_MODEL'], doc_chunk)
            for doc_chunk in doc_chunks
        ]
        self._grounding = GPTGrounding(os.environ['GPT_MODEL'])

    def call(self, query):
        worker_outputs = [worker.ask(query) for worker in self._workers]
        logging.debug(f"Grounding...")
        return self._grounding.ask(query, worker_outputs)
