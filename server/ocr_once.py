import sys
import json
import os
from pathlib import Path

import pandas as pd

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorDevice, AcceleratorOptions
from docling_core.types.doc import PictureItem, ImageRefMode

# ==========================================================
# [설정] 속도 최적화 (CUDA 사용)
# ==========================================================
FAST_MODE = True  # True여도 GPU를 쓰면 OCR이 빠릅니다.
IMAGE_RESOLUTION_SCALE = float(os.getenv("DL_IMAGE_SCALE", "1.5"))  # 이미지 해상도 (2.0 권장)
MAX_TABLE_PREVIEW_ROWS = int(os.getenv("DL_MAX_TABLE_PREVIEW_ROWS", "5"))


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)[:120]


def build_docling_converter():
    pdf_opts = PdfPipelineOptions()
    
    # 1. GPU 가속 활성화
    pdf_opts.accelerator_options = AcceleratorOptions(
        num_threads=8, 
        device=AcceleratorDevice.CUDA
    )

    # 2. 이미지 스케일 및 추출 설정
    pdf_opts.images_scale = IMAGE_RESOLUTION_SCALE
    pdf_opts.generate_page_images = False  # 전체 페이지 이미지는 끔 (용량 절약)
    pdf_opts.generate_picture_images = True # 개별 그림/표 이미지는 킴 (필수)

    # 3. OCR 설정 (GPU가 있으므로 OCR 켜도 빠릅니다)
    # 스캔된 PDF도 처리하려면 do_ocr=True가 좋습니다.
    pdf_opts.do_ocr = True 
    pdf_opts.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
        }
    )
    return converter


def extract_with_docling(input_path: Path, out_dir: Path):
    """
    Docling으로:
    - 전체 텍스트(Markdown)
    - 표들(DataFrame → rows/preview_rows)
    - 그림(PictureItem) 이미지 파일 JPEG 저장
    를 추출하고 메타데이터를 JSON-friendly dict로 리턴.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 컨버터 빌드 (GPU 설정 적용됨)
    converter = build_docling_converter()
    
    # 변환 시작
    conv_res = converter.convert(str(input_path))
    doc = conv_res.document

    stem = safe_name(input_path.stem)

    # -------- 1) 텍스트 (Markdown) --------
    try:
        text_md = doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
    except TypeError:
        text_md = doc.export_to_markdown()

    # -------- 2) 표 (DataFrame + 이미지) --------
    tables_meta = []
    table_image_idx = 0

    for idx, table in enumerate(doc.tables, start=1):
        # DataFrame 변환
        try:
            df: pd.DataFrame = table.export_to_dataframe(doc=doc)
        except TypeError:
            df: pd.DataFrame = table.export_to_dataframe()

        rows = df.values.tolist()
        preview_rows = df.head(MAX_TABLE_PREVIEW_ROWS).values.tolist()
        headers = list(df.columns.astype(str))

        # caption / page 추출
        caption = None
        if hasattr(table, "caption_text"):
            c = table.caption_text(doc=doc) if callable(table.caption_text) else table.caption_text
            caption = c

        page_no = None
        prov = getattr(table, "prov", [])
        if prov:
            page_no = getattr(prov[0], "page_no", None)

        # 표 이미지 (TableItem → 이미지 추출, JPEG로 저장)
        img_path = None
        try:
            img = table.get_image(doc)
            table_image_idx += 1
            img_filename = f"{stem}-table-{table_image_idx:03d}.jpg"
            img_full = out_dir / img_filename
            img.save(img_full, "JPEG", quality=70)
            img_path = str(img_full)
        except Exception:
            img_path = None  # 이미지가 없거나 실패한 경우

        tables_meta.append({
            "source": "docling",
            "index": idx,
            "page": int(page_no) if page_no is not None else None,
            "caption": caption,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "header": headers,
            "rows": rows,
            "preview_rows": preview_rows,
            "image_path": img_path,
        })

    # -------- 3) 그림(PictureItem) 이미지 --------
    pictures_meta = []
    picture_idx = 0

    for element, _level in doc.iterate_items():
        if not isinstance(element, PictureItem):
            continue

        picture_idx += 1
        prov = getattr(element, "prov", [])
        page_no = getattr(prov[0], "page_no", None) if prov else None

        caption = None
        if hasattr(element, "caption_text"):
            c = element.caption_text(doc=doc) if callable(element.caption_text) else element.caption_text
            caption = c

        img_path = None
        try:
            img = element.get_image(doc)
            pic_filename = f"{stem}-picture-{picture_idx:03d}.jpg"
            img_full = out_dir / pic_filename
            img.save(img_full, "JPEG", quality=70)
            img_path = str(img_full)
        except Exception:
            img_path = None

        pictures_meta.append({
            "source": "docling",
            "index": picture_idx,
            "page": int(page_no) if page_no is not None else None,
            "caption": caption,
            "image_path": img_path,
        })

    return {
        "ok": True,
        "engine": "docling",
        "file": str(input_path),
        "text": text_md,
        "tables": tables_meta,
        "pictures": pictures_meta,
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: python ocr_once.py <path> [out_dir]"}, ensure_ascii=False))
        return

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(json.dumps({"ok": False, "error": f"file not found: {input_path}"}, ensure_ascii=False))
        return

    if len(sys.argv) >= 3:
        out_dir = Path(sys.argv[2])
    else:
        out_dir = input_path.parent / f"{input_path.stem}_assets"

    try:
        result = extract_with_docling(input_path, out_dir)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": f"docling failed: {repr(e)}"
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()