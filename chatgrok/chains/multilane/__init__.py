import os
import logging
import threading
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
        threads = []
        worker_outputs = []

        # Start threads
        for index, worker in enumerate(self._workers):
            logger.debug("Starting worker thread [%d]", index)

            def worker_thread_function():
                return worker_outputs.append(worker.ask(query))
            worker_thread = threading.Thread(target=worker_thread_function)
            threads.append(worker_thread)
            worker_thread.start()

        # Join threads
        for index, thread in enumerate(threads):
            logger.debug("Joining worker thread [%d]", index)
            thread.join()
            logger.info("Worker thread [%d] is done", index)

        logger.info(f"Grounding...")
        return self._grounding.ask(query, worker_outputs)
