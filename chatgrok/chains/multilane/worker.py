import logging
from langchain.prompts import PromptTemplate
import openai

logger = logging.getLogger('chatgrok.multilane.worker')


class GPTWorker:
    def __init__(self, model, doc_chunk, prompt_loader):
        self._doc_chunk = doc_chunk
        self._model = model
        self._prompt_loader = prompt_loader
        system_preamble = self._prompt_loader.load(
            'worker.system_preamble',
            required=False
        )
        self._chat_history = [
            {
                'role': 'system',
                'content': PromptTemplate(
                    template=system_preamble,
                    input_variables=['doc_chunk']
                ).format(doc_chunk=doc_chunk.page_content)
            }
        ] if system_preamble else []

    def ask(self, query):
        user_prompt = self._prompt_loader.load('worker.ask')
        template_input_variables = {'query': query}
        if '{doc_chunk}' in user_prompt:
            template_input_variables['doc_chunk'] = self._doc_chunk
        self._chat_history.append({
            'role': 'user',
            'content': PromptTemplate(
                template=user_prompt,
                input_variables=list(template_input_variables.keys())
            ).format(**template_input_variables)
        })
        if logger.isEnabledFor(logging.DEBUG):
            import json
            logger.debug(json.dumps(self._chat_history, indent=4))
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=self._chat_history
        )
        assistant_response_message = completion.choices[0].message
        self._chat_history.append(assistant_response_message)
        return assistant_response_message['content']
