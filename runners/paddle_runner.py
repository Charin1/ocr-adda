# runners/paddle_runner.py

import tempfile
from pathlib import Path
try:
    from paddleocr import PPStructureV3, PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

class PaddleStructureRunner:
    name = "paddle_ppstructure"
    def __init__(self):
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddleOCR/PPStructure not installed")
        
        # Try to initialize the full structure pipeline first
        try:
            # --- THE FIX IS HERE ---
            # PPStructureV3 requires the OLD 'use_gpu' argument.
            self.pipeline = PPStructureV3(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False
            )
            self.use_structure = True
        except Exception:
            # If PPStructure fails, fallback to the standard PaddleOCR
            # --- THE FIX IS ALSO HERE ---
            # The base PaddleOCR class requires the NEW 'use_device' argument.
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en'
            )
            self.use_structure = False
            
    def run_image(self, img_path_or_array):
        if self.use_structure:
            # This part uses the self.pipeline (PPStructure)
            res_iter = self.pipeline.predict_iter([img_path_or_array])
            for res in res_iter:
                try:
                    md = res.get_text()
                    if md:
                        return md
                except Exception:
                    pass
                try:
                    tmp_md = Path(tempfile.gettempdir()) / "pp_tmp.md"
                    res.save_to_markdown(str(tmp_md))
                    return tmp_md.read_text(encoding='utf-8')
                except Exception:
                    return ""
            return ""
        else:
            # This part uses the self.ocr (standard OCR)
            raw = self.ocr.ocr(str(img_path_or_array), cls=True)
            texts = []
            if not raw:
                return ""

            flat_results = raw[0] if (isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list)) else raw
            
            if flat_results is None:
                return ""

            for ent in flat_results:
                try:
                    rec = ent[1]
                    if isinstance(rec, (list, tuple)):
                        texts.append(rec[0])
                    else:
                        texts.append(str(rec))
                except (TypeError, IndexError):
                    texts.append(str(ent))
            return "\n".join(texts)