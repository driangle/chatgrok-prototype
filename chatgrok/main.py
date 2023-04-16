import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

from .loader import PDFLoader
from .chains.multilane import Multilane


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
    # Configure OpenAI
    openai.organization = os.environ['OPENAPI_ORG_ID']
    openai.api_key = os.environ['OPENAPI_API_KEY']

    loader = PDFLoader()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    # Ingest / Load
    logging.info(f"Loading PDF File [{args.pdf_filepath}]...")
    pdf_doc = loader.load(args.pdf_filepath)
    logging.info(
        f"Successfully loaded PDF File [{args.pdf_filepath}], length: [{len(pdf_doc.page_content)}]"
    )
    # Ingest / Split
    logging.info("Splitting PDF File...")
    doc_chunks = splitter.split_documents([pdf_doc])
    logging.info(f"Split document into [{len(doc_chunks)}] chunks")

    # Query Loop
    chain = Multilane(doc_chunks)
    while True:
        try:
            logging.info("Ask a Question:\n")
            query = input()
            if not query or not query.strip():
                continue
            logging.info("Thinking...\n")
            response = chain.call(query)
            logging.info(f"\nAnswer:\n\n\t{response}\n\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
