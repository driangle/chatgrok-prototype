import logging
from langchain.prompts import PromptTemplate
import openai

logger = logging.getLogger('chatgrok.multilane.grounding')

class GPTGrounding:
    def __init__(self, model, prompt_loader):
        self._model = model
        self._prompt_loader = prompt_loader
        system_preamble = self._prompt_loader.load(
            'grounding.system_preamble',
            required=False
        )
        self._chat_history = [
            {
                'role': 'system',
                'content': PromptTemplate(
                    template=system_preamble,
                    input_variables=[]
                ).format()
            }
        ] if system_preamble else []

    def ask(self, query, worker_outputs):
        ask_template = self._prompt_loader.load('grounding.ask')
        template_input_variables = {
            'worker_outputs': '\n\n'.join(worker_outputs)}
        if '{query}' in ask_template:
            template_input_variables['query'] = query
        self._chat_history.append({
            'role': 'user',
            'content': PromptTemplate(
                template=ask_template,
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
