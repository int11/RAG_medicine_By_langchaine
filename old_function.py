import fitz  # PyMuPDF

# 문서 형식화 함수 정의
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        texts.append(page.get_text())
    return texts


# 간단한 키워드 기반 문서 검색기 정의
class SimpleKeywordRetriever:
    def __init__(self, documents):
        self.documents = documents
   
    def retrieve(self, query):
        # 키워드 기반 검색
        results = [doc for doc in self.documents if query in doc.page_content]
        return results
