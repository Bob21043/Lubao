import fitz
import re
from typing import List
from langchain_core.documents import Document

def clean_pdf_text(raw_text: str) -> str:
    """清理 PDF 文字雜訊"""
    text = re.sub(r'圖[\u2013-]?\d+', '', raw_text)
    text = re.sub(r'（見圖[\u2013-]?\d+）', '', text)
    text = re.sub(r'[ \t\u3000]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_pdf_by_visual_chapter(pdf_path: str, chapter_font_size: float = 15.96) -> List[tuple[str, str]]:
    """根據字體大小切章節，回傳 (標題, 章節全文) 清單"""
    doc = fitz.open(pdf_path)
    chapters = []
    chapter_titles = []
    current_chapter = ""
    current_title = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    size = round(span["size"], 2)
                    if size == round(chapter_font_size, 2) and re.match(r'^\d+\.\d+', text):
                        if current_chapter:
                            chapters.append((current_title, current_chapter.strip()))
                        current_title = text
                        chapter_titles.append(current_title)
                        current_chapter = text + "\n"
                    else:
                        current_chapter += text + "\n"

    if current_chapter:
        chapters.append((current_title, current_chapter.strip()))

    print(f"\U0001F4D8 找到章節數：{len(chapter_titles)}")
    return chapters

def convert_to_documents(chapters: List[tuple[str, str]]) -> List[Document]:
    """將 (標題,內容) 清單轉換為 Document 物件"""
    return [
        Document(page_content=clean_pdf_text(content), metadata={"title": title})
        for title, content in chapters
    ]

def convert_to_chunk(documents: List[Document], chunk_size: int = 500) -> List[Document]:
    """將每個章節 Document 切成固定長度段落並加上 chunk_id"""
    chunked_docs = []
    for doc in documents:
        content = doc.page_content
        title = doc.metadata.get("title", "未知章節")
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        print(f"\U0001F539 {title} 切成 {len(chunks)} 段")
        for idx, chunk in enumerate(chunks):
            chunked_docs.append(Document(
                page_content=chunk,
                metadata={"title": title, "chunk_id": idx}
            ))
    return chunked_docs