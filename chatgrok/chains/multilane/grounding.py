import logging
from langchain.prompts import PromptTemplate
import openai

class GPTGrounding:
    def __init__(self, model, prompt_loader):
        self._model = model
        self._prompt_loader = prompt_loader

    def ask(self, query, worker_outputs):
        system_preamble = self._prompt_loader.load(
            'grounding.system_preamble', 
            required=False
        )
        messages = [
            {
                'role': 'system',
                'content': PromptTemplate(
                    template=system_preamble,
                    input_variables=[]
                ).format()
            }
        ] if system_preamble else []
        messages.append({
            'role': 'user',
            'content': PromptTemplate(
                template=self._prompt_loader.load('grounding.ask'),
                input_variables=['worker_outputs', 'query']
            ).format(worker_outputs='\n\n'.join(worker_outputs), query=query)
        })

        # messages.append({
        #     'role': 'user',
        #     'content': '\n'.join([f'\n\nDOCUMENT:\n{worker_output}\n\n'
        #                           for worker_output in worker_outputs
        #                           ])
        # })
        # messages.append({
        #     'role': 'user',
        #     'content': f'QUESTION: {query}'
        # })
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            import json
            print(json.dumps(messages, indent=4))
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=messages
        )
        assistant_response_message = completion.choices[0].message
        return assistant_response_message['content']
