from langchain.document_loaders import UnstructuredPDFLoader

class PDFLoader:

    def load(self, filename):
        loader = UnstructuredPDFLoader(filename)
        return loader.load()