import openai

class GPTWorker:
    def __init__(self, model, doc_chunk):
        self._model = model
        self._doc_chunk = doc_chunk
        self._chat_history = [
            {
                'role': 'system',
                'content': '\n'.join([
                    'Answer all user questions using the text provider. Limit your response to 512 tokens.',
                    f'TEXT: {doc_chunk}'
                ])
            }
        ]

    def ask(self, query):
        self._chat_history.append({
            'role': 'user',
            'content': '\n'.join([
                query
                # f"Answer the question",
                # 'Provide just 'non relevant information' as a response if text does not contain any relevant information. Limit your response to 512 tokens.
            ])
        })
        # logging.debug(self._chat_history)
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=self._chat_history
        )
        assistant_response_message = completion.choices[0].message
        self._chat_history.append(assistant_response_message)
        return assistant_response_message['content']
