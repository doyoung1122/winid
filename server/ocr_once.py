import sys
import json
from pathlib import Path
import os
OCR_LANGS = os.getenv("OCR_LANGS", "kor+eng").split("+")
PDF_INFER_TABLES = os.getenv("PDF_INFER_TABLES", "true").lower() == "true"

def extract_with_unstructured(pdf_path: Path):
    """
    1) unstructured로 표 먼저 시도
       - 우선 hi_res (가능한 경우)
       - 실패 시 auto 로 재시도
    2) 본문 텍스트도 함께 추출
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except Exception as e:
        return {"ok": False, "error": f"unstructured import failed: {e}"}

    tables = []
    text_chunks = []

    # 시도 순서: hi_res -> auto
    strategies = []
    # hi_res는 Windows에서 의존성 부족할 수 있음
    strategies.append(("hi_res", True))
    strategies.append(("auto", False))

    last_err = None
    for strategy, prefers_hi_res in strategies:
        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                infer_table_structure=True if PDF_INFER_TABLES else False,
                strategy=strategy,
                languages=[lang for lang in OCR_LANGS if lang],  # ["kor","eng"] 등
            )
            # 성공 시 파싱
            for el in elements:
                cat = getattr(el, "category", None)
                # 본문 텍스트
                if getattr(el, "text", None):
                    text_chunks.append(el.text)

                # 표만 수집
                if cat == "Table":
                    meta = getattr(el, "metadata", None)
                    page_num = getattr(meta, "page_number", None) if meta else None

                    # text_as_html 같은 속성은 버전에 따라 없을 수 있어 안전 처리
                    html = getattr(meta, "text_as_html", None) if meta else None
                    # 일부 버전은 el.metadata.to_dict() 지원
                    meta_dict = None
                    try:
                        meta_dict = meta.to_dict() if meta and hasattr(meta, "to_dict") else None
                    except Exception:
                        meta_dict = None

                    # 미리보기 행/열은 라이브러리 버전마다 접근법이 달라서 보수적으로 구성
                    tables.append({
                        "source": "unstructured",
                        "page": page_num,
                        "html": html,
                        "metadata": meta_dict,
                        # 아래 필드는 향후 확장용(없을 수 있음)
                        "text": getattr(el, "text", None),
                    })

            return {
                "ok": True,
                "text": "\n".join([t for t in text_chunks if t]),
                "tables": tables,
                "engine": f"unstructured:{strategy}",
            }

        except Exception as e:
            last_err = e
            # 다음 전략으로 폴백
            continue

    return {
        "ok": False,
        "error": f"unstructured failed on all strategies (last: {last_err})",
        "text": "",
        "tables": [],
    }


def extract_with_pdfplumber(pdf_path: Path, max_preview_rows=3):
    """
    pdfplumber 폴백: 각 페이지에서 표 감지 → 셀 텍스트를 2D 리스트로 수집
    """
    import pdfplumber

    out_tables = []
    text_all = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # 페이지 텍스트(본문)
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text:
                text_all.append(page_text)

            # 표 감지
            try:
                # 기본 테이블 셋팅(필요시 조정)
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                    }
                )
            except Exception:
                tables = []

            for tbl in tables or []:
                # tbl은 2D list (행렬)
                n_rows = len(tbl)
                n_cols = max((len(r) for r in tbl), default=0)

                # 미리보기(앞쪽 몇 행)
                preview = tbl[:max_preview_rows]

                out_tables.append({
                    "source": "pdfplumber",
                    "page": i,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "preview_rows": preview,
                    # 전체 테이블을 원하면 preview 대신 아래 주석을 쓰세요(용량 주의)
                    # "rows": tbl,
                })

    return {
        "ok": True,
        "text": "\n".join([t for t in text_all if t]),
        "tables": out_tables,
        "engine": "pdfplumber",
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: python ocr_once.py <pdf_path>"}))
        return

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(json.dumps({"ok": False, "error": f"file not found: {pdf_path}"}))
        return

    # 1) unstructured 먼저
    ures = extract_with_unstructured(pdf_path)
    if ures.get("ok") and ures.get("tables"):
        print(json.dumps(ures, ensure_ascii=False))
        return

    # 2) unstructured 성공했지만 표가 비어있거나, 실패한 경우 → pdfplumber 폴백
    pres = extract_with_pdfplumber(pdf_path)
    if pres.get("ok"):
        # unstructured에서 텍스트를 어느 정도 뽑았으면 합치기(중복 최소화)
        base_text = ures.get("text", "") if isinstance(ures, dict) else ""
        merged_text = (base_text + "\n" + pres.get("text", "")).strip() or pres.get("text", "")
        out = {
            "ok": True,
            "engine": f"{ures.get('engine', 'unstructured:failed')} -> {pres.get('engine')}",
            "text": merged_text,
            "tables": pres.get("tables", []),
        }
        print(json.dumps(out, ensure_ascii=False))
        return

    # 둘 다 실패
    print(json.dumps({
        "ok": False,
        "error": f"both extractors failed: unstructured={ures.get('error')}, pdfplumber failed",
        "text": "",
        "tables": [],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()