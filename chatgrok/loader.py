from langchain.document_loaders import UnstructuredPDFLoader

class PDFLoader:

    def load(self, filename):
        loader = UnstructuredPDFLoader(filename)
        docs = loader.load()
        if len(docs) != 1:
            raise Exception(f"Expected one PDF Document, got [{len(docs)}]")
        return docs[0]