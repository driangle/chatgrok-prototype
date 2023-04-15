import os
import logging
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai


class PDFLoader:

    def load(self, filename):
        loader = UnstructuredPDFLoader(filename)
        docs = loader.load()
        if len(docs) != 1:
            raise Exception(f"Expected one PDF Document, got [{len(docs)}]")
        return docs[0]


class GPTWorker:
    def __init__(self, model, doc_chunk):
        self._model = model
        self._doc_chunk = doc_chunk
        self._chat_history = [
            {
                'role': 'system',
                'content': f'Answer all user questions using the text provider\n. TEXT: {doc_chunk}'
            }
        ]

    def ask(self, query):
        self._chat_history.append({
            'role': 'user',
            'content': query
        })
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=self._chat_history
        )
        assistant_response_message = completion.choices[0].message
        self._chat_history.append(assistant_response_message)
        return assistant_response_message['content']


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
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=self._chat_history
        )
        assistant_response_message = completion.choices[0].message
        self._chat_history.append(assistant_response_message)
        # self._chat_history = []
        return assistant_response_message['content']


def configure_logging():
    logging.basicConfig(
        format="[%(levelname)s] %(name)s - %(message)s",
        level='INFO',
    )
    logging.getLogger('unstructured_inference').setLevel(logging.ERROR)
    logging.getLogger('detectron2').setLevel(logging.ERROR)
    logging.getLogger('fvcore').setLevel(logging.ERROR)


def main():
    load_dotenv()
    configure_logging()
    loader = PDFLoader()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    openai.organization = os.environ['OPENAPI_ORG_ID']
    openai.api_key = os.environ['OPENAPI_API_KEY']
    pdf_filepath = "./docs/Comet_Half-Year-Report-2022-short/Comet_Half-Year-Report-2022-short.pdf"
    logging.info(f"Loading PDF File [{pdf_filepath}]...")
    # pdf_file = loader.load("/Users/driangle/workplace/gg/books/dalle/dalle_2_prompt_book_1.0.2.pdf")
    # pdf_doc = loader.load("/Users/driangle/Downloads/pml.howto.pdf")
    pdf_doc = loader.load(pdf_filepath)
    logging.info(
        f"Successfully loaded PDF File [{pdf_filepath}], length: [{len(pdf_doc.page_content)}]")

    logging.info("Splitting PDF File...")
    doc_chunks = splitter.split_documents([pdf_doc])

    logging.info(f"Split document into [{len(doc_chunks)}] chunks")

    workers = [
        GPTWorker(os.environ['GPT_MODEL'], doc_chunk)
        for doc_chunk in doc_chunks
    ]
    grounding = GPTGrounding(os.environ['GPT_MODEL'])
    while True:
        try:
            logging.info("Ask a Question:\n")
            query = input()
            logging.info("Thinking...\n")
            logging.debug(f"Running [{len(workers)}] workers...")
            worker_outputs = [worker.ask(query) for worker in workers]
            logging.debug(f"Grounding...")
            response = grounding.ask(query, worker_outputs)
            logging.info(f"\nAnswer:\n\n\t{response}\n\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
