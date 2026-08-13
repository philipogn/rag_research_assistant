from pathlib import Path

from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel, _backend_to_engine_type, _rapidocr_artifacts, _resolve_rapidocr
from docling.utils.model_downloader import download_models

models_dir = Path("models/docling")

_RAPIDOCR_BACKEND = "onnxruntime"
_RAPIDOCR_LANG = "chinese"  # RapidOCR's default bucket - a general multilingual model, not Chinese-only


def _ensure_rapidocr_models() -> None:
    """Docling's own RapidOCR downloader (RapidOcrModel.download_models) fetches from a
    modelscope.cn CDN that isn't always reachable. If it fails, fall back to the `rapidocr`
    package's own downloader (a different, working source) and copy the files into the
    layout artifacts_path expects
    """
    rapidocr_dir = models_dir / RapidOcrModel._model_repo_folder
    resolved = _resolve_rapidocr(_RAPIDOCR_LANG, _RAPIDOCR_BACKEND)
    artifacts = _rapidocr_artifacts(
        rapidocr_dir,
        _backend_to_engine_type(_RAPIDOCR_BACKEND),
        resolved.ppocr_version,
        resolved.rapidocr_lang_token,
    )
    expected = [dest for artifact in artifacts.values() for dest in artifact.files]

    if all(p.exists() for p in expected):
        print("RapidOCR models already present, skipping.")
        return

    try:
        RapidOcrModel.download_models(
            backend=_RAPIDOCR_BACKEND, lang=_RAPIDOCR_LANG, local_dir=rapidocr_dir, progress=True,
        )
        return
    except Exception as exc:
        print(f"Docling's RapidOCR download failed ({exc}); falling back to rapidocr's own downloader.")

    from rapidocr import RapidOCR
    import rapidocr as rapidocr_pkg

    RapidOCR()  # triggers rapidocr's own lazy download into its package dir
    src_dir = Path(rapidocr_pkg.__file__).parent / "models"

    for dest in expected:
        src = src_dir / dest.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        print(f"Copied {src} -> {dest}")


def main() -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Docling models into {models_dir} ...")
    download_models(
        models_dir,
        with_rapidocr=False,
        with_code_formula=False,  # unused: do_code_enrichment/do_formula_enrichment are off in extract.py
        with_picture_classifier=False,  # unused: do_picture_classification is off in extract.py
        progress=True,
    )
    _ensure_rapidocr_models()
    print("Done")


if __name__ == "__main__":
    main()
