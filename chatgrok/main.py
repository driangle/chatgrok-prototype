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
    parser.add_argument("--prompt_experiment", "-pe", required=False, type=str)
    parser.add_argument("--splitter_chunk_size", "-scs",
                        required=False, type=int)
    parser.add_argument("--splitter_chunk_overlap",
                        "-sco", required=False, type=int)
    return parser.parse_args(args)


def configure_logging(verbose):
    logging.basicConfig(
        format="[%(levelname)s] chatgrok - %(message)s",
        level=logging.INFO
        # level='DEBUG' if verbose else 'INFO',
    )
    logging.getLogger('chatgrok').setLevel(
        logging.DEBUG if verbose else logging.INFO
    )
    logging.getLogger('unstructured').setLevel(logging.ERROR)
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
    logger = logging.getLogger('main')
    # Configure OpenAI
    openai.organization = os.environ['OPENAPI_ORG_ID']
    openai.api_key = os.environ['OPENAPI_API_KEY']

    loader = PDFLoader()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.splitter_chunk_size or 4000,
        chunk_overlap=args.splitter_chunk_overlap or 200
    )

    # Ingest / Load
    logger.info(f"Loading PDF File [{args.pdf_filepath}]...")
    doc_chunks = loader.load(args.pdf_filepath)
    logger.info(
        f"Successfully loaded PDF File [{args.pdf_filepath}], initial chunks: [{len(doc_chunks)}], length: [{sum([len(c.page_content) for c in doc_chunks])}]"
    )
    # Ingest / Split
    logger.info("Splitting PDF File...")
    doc_chunks = splitter.split_documents(doc_chunks)
    logger.info(f"Split document into [{len(doc_chunks)}] chunks")

    # Query Loop
    chain = Multilane(doc_chunks, args.prompt_experiment)
    while True:
        try:
            logger.info("Ask a Question:\n")
            query = input()
            if not query or not query.strip():
                continue
            print()
            logger.info("Thinking...\n")
            response = chain.call(query)
            logger.info(f"\nAnswer:\n\n\t{response}\n\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
