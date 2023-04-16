import openai

class GPTGrounding:
    def __init__(self, model):
        self._model = model
        self._chat_history = [
            {
                'role': 'system',
                'content': '\n'.join([
                    'The user will give you a set of documents followed by a question.',
                    'Each time try to answer the question using all previously received documents'
                ])
            }
        ]

    def ask(self, query, worker_outputs):
        self._chat_history.append({
            'role': 'user',
            'content': '\n'.join([f'\n\nDOCUMENT:\n{worker_output}\n\n'
                                  for worker_output in worker_outputs
                                  ])
        })
        self._chat_history.append({
            'role': 'user',
            'content': f'QUESTION: {query}'
        })
        # logging.debug(self._chat_history)
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=self._chat_history
        )
        assistant_response_message = completion.choices[0].message
        self._chat_history.append(assistant_response_message)
        return assistant_response_message['content']

