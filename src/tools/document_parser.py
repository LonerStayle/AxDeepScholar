from docling.document_converter import DocumentConverter
import fitz, pdfplumber, layoutparser as lp, cv2, os

import numpy as np
os.environ["DOC_ACCELERATOR_DEVICE"] = "cpu"
os.environ["DOC_ACCELERATOR_BACKEND"] = "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["DOCLING_PICTURE_DESCRIPTIONS"] = "" 

# ——— 1️⃣ 텍스트 레이어 확인 ———
def has_text_layer(pdf_path: str) -> bool:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    for pg in reader.pages[:3]:
        txt = pg.extract_text()
        if txt and len(txt.strip()) > 50:
            return True
    return False

# ——— 2️⃣ 좌표 기반 텍스트 정렬 (PyMuPDF) ———
def extract_text_mupdf_clean_strict(pdf_path: str):
    """
    - 페이지 상/하/모서리/작은 폰트 제거
    - '붙어버린 단어'를 x-좌표 간격 기반으로 스페이스 복원
    - 두단 문서는 왼쪽 칼럼 전체 → 오른쪽 칼럼 전체 순서 유지 (A1 A2 A3 B1 B2 B3)
    """
    import fitz, numpy as np, re

    doc = fitz.open(pdf_path)
    all_pages = []

    for page in doc:
        page_w, page_h = page.rect.width, page.rect.height
        data = page.get_text("dict")
        if not data or "blocks" not in data:
            continue

        # 1) 라인 단위로 스팬을 모으면서, 페이지 여백/폰트 크기/본문 프레임으로 1차 필터
        line_items = []  # [(y_top, x_min, line_text), ...]
        for b in data["blocks"]:
            for line in b.get("lines", []):
                spans = []
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if not txt:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    fsz = span.get("size", 0)

                    # 상단/하단 10% 컷 + 모서리 컷 + 너무 작은 폰트 컷
                    if y0 < page_h * 0.20 or y1 > page_h * 0.80:
                        continue
                    if x0 < page_w * 0.10 or x1 > page_w * 0.90:
                        continue
                    if fsz < 8:
                        continue

                    spans.append((x0, y0, x1, y1, fsz, txt))

                if not spans:
                    continue

                # 2) 같은 라인의 스팬들을 x0 기준 정렬하고, 간격으로 스페이스 복원
                spans.sort(key=lambda s: s[0])  # x0
                rebuilt_parts = []
                prev_x1 = None
                prev_fsz = None

                for x0, y0, x1, y1, fsz, txt in spans:
                    # 띄어쓰기 복원 휴리스틱
                    # - 스팬 간 gap > max(0.5*font_size, 2.0) 이면 공백 삽입
                    # - 단어 끝이 하이픈으로 끊긴 경우는 공백 삽입하지 않음 (줄바꿈 연결)
                    gap = 0 if prev_x1 is None else (x0 - prev_x1)
                    need_space = (prev_x1 is not None) and (gap > max(0.5 * (prev_fsz or fsz), 2.0))

                    if need_space and not (rebuilt_parts and rebuilt_parts[-1].endswith("-")):
                        rebuilt_parts.append(" ")

                    rebuilt_parts.append(txt)
                    prev_x1 = x1
                    prev_fsz = fsz

                line_text = "".join(rebuilt_parts).strip()

                # 라인 끝 하이픈 처리: 줄바꿈 하이픈으로 이어진 경우 단어 붙이기
                # e.g., "investi-" + next line "gation" → "investigation"
                line_text = re.sub(r"-\s*$", "", line_text)

                # 라인의 대표 좌표(위쪽 y, 최소 x) 저장
                y_top = min(s[1] for s in spans)
                x_min = min(s[0] for s in spans)

                # 너무 짧은 잡음 라인은 스킵
                if len(line_text) < 3:
                    continue
                line_items.append((y_top, x_min, line_text))

        if not line_items:
            continue

        # 3) 좌/우 칼럼 분리 후, "왼쪽 전부 → 오른쪽 전부" 순서 유지
        #    (요청하신 A1 A2 A3 B1 B2 B3 순서)
        mid_x = page_w / 2.0
        left_lines  = [(y, x, t) for (y, x, t) in line_items if x < mid_x]
        right_lines = [(y, x, t) for (y, x, t) in line_items if x >= mid_x]

        left_lines.sort(key=lambda r: (round(r[0], 1), round(r[1], 1)))
        right_lines.sort(key=lambda r: (round(r[0], 1), round(r[1], 1)))

        # 4) 라인 합치기 + 공백/중복 정리
        def join_lines(lines):
            # 라인 사이 불필요한 중복 공백 정리
            text = "\n".join(t for (_, _, t) in lines)
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

        page_text = "\n".join([
            join_lines(left_lines),
            join_lines(right_lines)
        ]).strip()

        # 너무 빈약한 페이지는 제외
        if len(page_text) >= 30:
            all_pages.append(page_text)

    return "\n\n".join(all_pages)

# def extract_text_mupdf_clean_strict(pdf_path: str):
#     """
#     두단(column) 문서는 그대로 왼쪽→오른쪽 순서로 읽되,
#     본문 바깥(상단/하단/모서리/작은 폰트) 잡음은 제거.
#     """
#     import fitz
#     import numpy as np

#     doc = fitz.open(pdf_path)
#     pages = []
#     header_texts = set()
#     footer_texts = set()

#     # 1️⃣ 헤더/푸터 후보 텍스트 추정
#     for page in doc[:min(3, len(doc))]:
#         blocks = page.get_text("blocks")
#         blocks = sorted(blocks, key=lambda b: b[1])  # y좌표
#         if not blocks:
#             continue
#         top_text = blocks[0][4].strip()
#         bottom_text = blocks[-1][4].strip()
#         if len(top_text) < 120:
#             header_texts.add(top_text)
#         if len(bottom_text) < 120:
#             footer_texts.add(bottom_text)

#     # 2️⃣ 본문 파싱
#     for page in doc:
#         page_w, page_h = page.rect.width, page.rect.height
#         blocks = page.get_text("dict")["blocks"]
#         filtered_blocks = []

#         # 🔹 블록 단위로 필터링 (폰트크기 + 좌표)
#         for b in blocks:
#             for line in b.get("lines", []):
#                 for span in line.get("spans", []):
#                     text = span["text"].strip()
#                     if not text:
#                         continue
#                     x0, y0, x1, y1 = span["bbox"]
                    
#                     font_size = span.get("size", 0)
#                     if y0 < page_h * 0.15 or y1 > page_h * 0.85:
#                         continue

#                     # 너무 작은 폰트는 제거 (각주, 학회명 등)
#                     if font_size < 8:
#                         continue

#                     # 페이지 바깥쪽 잡영역 (상단·하단·모서리)
#                     margin_x_ratio = (x0 / page_w, x1 / page_w)
#                     margin_y_ratio = (y0 / page_h, y1 / page_h)
#                     if margin_y_ratio[0] < 0.06 or margin_y_ratio[1] > 0.94:
#                         continue
#                     if margin_x_ratio[0] < 0.04 or margin_x_ratio[1] > 0.96:
#                         continue

#                     filtered_blocks.append((x0, y0, x1, y1, text))

#         # 두단 감지 및 정렬 (기존 방식 그대로 유지)
#         if not filtered_blocks:
#             continue
#         widths = [b[2]-b[0] for b in filtered_blocks]
#         median_width = np.median(widths)
#         two_column = median_width < page_w * 0.45

#         left_blocks = [b for b in filtered_blocks if b[0] < page_w / 2]
#         right_blocks = [b for b in filtered_blocks if b[0] >= page_w / 2]

#         def sort_blocks(blks):
#             return sorted(blks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

#         lines = []
#         for blk in sort_blocks(left_blocks) + sort_blocks(right_blocks):
#             lines.append(blk[4].strip())

#         text = "\n".join(lines)
#         if len(text.strip()) > 30:
#             pages.append(text)

#     return "\n\n".join(pages)

# ——— 3️⃣ 표 및 세밀한 줄 보정 (pdfplumber) ———
def extract_text_pdfplumber(pdf_path: str):
    all_text = []
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            chars = sorted(page.chars, key=lambda c: (round(c["top"], 1), c["x0"]))
            prev_y = None
            txt = ""
            for ch in chars:
                if prev_y is None or abs(ch["top"] - prev_y) > 5:
                    txt += "\n"
                txt += ch["text"]
                prev_y = ch["top"]
            all_text.append(txt)
            for tbl in page.extract_tables():
                # 표를 마크다운 등 문자열로 저장
                import pandas as pd
                df = pd.DataFrame(tbl[1:], columns=tbl[0])
                tables.append({"page": i, "content": df.to_markdown(index=False)})
    return "\n\n".join(all_text), tables


# ——— 4️⃣ 페이지별 구조 감지 (텍스트 vs 이미지 vs 표 등) ———
def analyze_page_type(pdf_path):
    """페이지별로 텍스트/이미지/표 비율 분석"""
    doc = fitz.open(pdf_path)
    page_types = []
    for i, page in enumerate(doc):
        img_count = len(page.get_images())
        text_blocks = page.get_text("blocks")
        text_len = sum(len(b[4].strip()) for b in text_blocks if b[4].strip())
        text_lower = (page.get_text("text") or "").lower()
        
        if img_count > 0 and text_len > 400:
            ptype = "text+image"
        elif img_count > 0 and text_len < 100:
            ptype = "image_only"
        elif "table" in text_lower or "표" in text_lower:
            ptype = "text+table"
        else:
            ptype = "text_only"

        page_types.append({"page": i, "type": ptype, "img_count": img_count, "text_len": text_len})
    return page_types


docling = DocumentConverter()
def parse_docling_only(pdf_path: str, page_numbers: list[int] | None = None):
    """
    최신 Docling API에서는 page_numbers 인자를 지원하지 않음.
    따라서 전체 문서를 변환하고 이후 필요한 페이지만 필터링.
    """
    result = docling.convert(pdf_path)
    doc_struct = result.document

    # page_numbers가 있을 경우 — 해당 페이지 데이터만 필터링
    if page_numbers is not None and hasattr(doc_struct, "sections"):
        selected_sections = [
            sec for sec in doc_struct.sections
            if getattr(sec, "page_number", None) in page_numbers
        ]
        # 임시 구조체 흉내 — section만 교체
        doc_struct.sections = selected_sections

    return doc_struct
# ——— 5️⃣ 페이지별로 파서 선택 실행 ———
def parse_page_adaptively(pdf_path):
    """페이지 단위 적응형 파싱 (빈 페이지 누락 방지)"""
    page_summaries = analyze_page_type(pdf_path)
    results = []

    import pandas as pd

    for info in page_summaries:
        page_idx = info["page"]
        ptype = info["type"]
        base_meta = {"page": page_idx, "source": os.path.basename(pdf_path)}

        try:
            if ptype == "text_only":
                # ✅ Docling section이 없더라도 mupdf fallback 추가
                doc_struct = parse_docling_only(pdf_path, page_numbers=[page_idx + 1])
                section_added = False
                for sec in getattr(doc_struct, "sections", []):
                    results.append({
                        "page_content": sec.text.strip(),
                        "metadata": {**base_meta, "type": "text"}
                    })
                    section_added = True
                if not section_added:
                    # 🔻 특정 페이지만 추출
                    doc = fitz.open(pdf_path)
                    page = doc[page_idx]
                    txt = page.get_text("text")
                    results.append({
                        "page_content": txt.strip(),
                        "metadata": {**base_meta, "type": "text-fallback"}
                    })

            elif ptype == "text+table":
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[page_idx]
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    if tables:
                        for tbl in tables:
                            df = pd.DataFrame(tbl[1:], columns=tbl[0])
                            results.append({
                                "page_content": df.to_markdown(index=False),
                                "metadata": {**base_meta, "type": "table"}
                            })
                    if text.strip():
                        results.append({
                            "page_content": text.strip(),
                            "metadata": {**base_meta, "type": "text+table"}
                        })
                    else:
                        # ✅ 표만 있고 텍스트 없는 경우라도 dummy 텍스트 추가
                        results.append({
                            "page_content": "[No text on this page]",
                            "metadata": {**base_meta, "type": "table-only"}
                        })
            
            elif ptype in ["text+image","image_only"] :
                doc_struct = parse_docling_only(pdf_path, page_numbers=[page_idx + 1])
                section_added = False
                for sec in getattr(doc_struct, "sections", []):
                    results.append({
                        "page_content": sec.text.strip(),
                        "metadata": {**base_meta, "type": "text"}
                    })
                    section_added = True
                if not section_added:
                    # 특정 페이지만 추출
                    doc = fitz.open(pdf_path)
                    page = doc[page_idx]
                    txt = page.get_text("text")
                    results.append({
                        "page_content": txt.strip(),
                        "metadata": {**base_meta, "type": "text-fallback"}
                    })
        except Exception as e:
            #  페이지별 에러 무시하고 계속 진행
            results.append({
                "page_content": f"[Page {page_idx} skipped due to error: {str(e)}]",
                "metadata": {**base_meta, "type": "error"}
            })
    return results


# ——— 메인 파싱 함수 (Docling 기반) ———
def parse_docling_with_fallback(pdf_path: str):
    """
    Docling을 메인 파서로 사용하되, fallback 및 보정 병합 로직 포함
    """
    docs = []
    

    if not has_text_layer(pdf_path):
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(pdf_path)
        for pi, img in enumerate(images):
            txt = pytesseract.image_to_string(img, lang="eng+kor")
            docs.append({
                "page_content": txt,
                "metadata": {"source": os.path.basename(pdf_path), "page": pi, "type": "ocr"}
            })
        return docs
    
    converted = parse_docling_only(pdf_path)
    doc_struct = converted.document

    for sec in doc_struct.sections:
        docs.append({
            "page_content": sec.text,
            "metadata": {"source": os.path.basename(pdf_path), "type": "section", "title": sec.title if hasattr(sec, "title") else None}
        })

    for tbl in doc_struct.tables:
        docs.append({
            "page_content": tbl.html or tbl.markdown or tbl.csv, 
            "metadata": {"source": os.path.basename(pdf_path), "type": "table", "page": tbl.page_number}
        })

    if not docs:
        t1 = extract_text_mupdf_clean_strict(pdf_path)
        t2, tbls = extract_text_pdfplumber(pdf_path)
        
        merged = t1 if len(t1) > len(t2) else t2
        docs.append({
            "page_content": merged,
            "metadata": {"source": os.path.basename(pdf_path), "type": "fallback"}
        })
        for tm in tbls:
            docs.append({
                "page_content": tm["content"],
                "metadata": {"source": os.path.basename(pdf_path), "type": "table", "page": tm["page"]}
            })

    return docs


# ——— 6️⃣ 전체 문서 파싱 컨트롤러 ———
def parse_with_docling(pdf_path: str):
    """
    PDF를 페이지별로 분석하여
    Docling / pdfplumber / layoutparser / OCR 을 자동 적용
    """
    docs = []

    if not has_text_layer(pdf_path):
        # 전체가 이미지 PDF일 경우 OCR
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(pdf_path)
        for pi, img in enumerate(images):
            txt = pytesseract.image_to_string(img, lang="eng+kor")
            docs.append({
                "page_content": txt,
                "metadata": {"source": os.path.basename(pdf_path), "page": pi, "type": "ocr"}
            })
        return docs

    # ✅ 페이지 단위 파싱 실행
    adaptive_docs = parse_page_adaptively(pdf_path)

    # ✅ fallback 안전장치 (혹시 실패했을 때)
    if not adaptive_docs:
        t1 = extract_text_mupdf_clean_strict(pdf_path)
        t2, tbls = extract_text_pdfplumber(pdf_path)
        merged = t1 if len(t1) > len(t2) else t2
        docs.append({
            "page_content": merged,
            "metadata": {"source": os.path.basename(pdf_path), "type": "fallback"}
        })
        for tm in tbls:
            docs.append({
                "page_content": tm["content"],
                "metadata": {"source": os.path.basename(pdf_path), "type": "table", "page": tm["page"]}
            })
    else:
        docs.extend(adaptive_docs)

    return docs