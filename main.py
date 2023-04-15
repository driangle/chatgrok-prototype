import os
import sys
import logging
import argparse
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


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--pdf_filepath", "-f", required=True, type=str)
    return parser.parse_args(args)


def configure_logging(verbose):
    logging.basicConfig(
        format="[%(levelname)s] %(name)s - %(message)s",
        level='DEBUG' if verbose else 'INFO',
    )
    logging.getLogger('unstructured_inference').setLevel(logging.ERROR)
    logging.getLogger('detectron2').setLevel(logging.ERROR)
    logging.getLogger('fvcore').setLevel(logging.ERROR)
    logging.getLogger('pdfminer').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('layoutparser').setLevel(logging.ERROR)


def main(args):
    # Setup
    load_dotenv()
    configure_logging(args.verbose)
    loader = PDFLoader()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    grounding = GPTGrounding(os.environ['GPT_MODEL'])
    openai.organization = os.environ['OPENAPI_ORG_ID']
    openai.api_key = os.environ['OPENAPI_API_KEY']

    # Ingest
    logging.info(f"Loading PDF File [{args.pdf_filepath}]...")
    pdf_doc = loader.load(args.pdf_filepath)
    logging.info(
        f"Successfully loaded PDF File [{args.pdf_filepath}], length: [{len(pdf_doc.page_content)}]"
    )

    logging.info("Splitting PDF File...")
    doc_chunks = splitter.split_documents([pdf_doc])

    logging.info(f"Split document into [{len(doc_chunks)}] chunks")

    # Build Workers
    workers = [
        GPTWorker(os.environ['GPT_MODEL'], doc_chunk)
        for doc_chunk in doc_chunks
    ]
    
    # Query Loop
    while True:
        try:
            logging.info("Ask a Question:\n")
            query = input()
            if not query or not query.strip():
                continue
            logging.info("Thinking...\n")
            logging.debug(f"Running [{len(workers)}] workers...")
            worker_outputs = [worker.ask(query) for worker in workers]
            logging.debug(f"Grounding...")
            response = grounding.ask(query, worker_outputs)
            logging.info(f"\nAnswer:\n\n\t{response}\n\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
