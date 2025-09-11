import easyocr
class EasyOCRRunner:
    name = "easyocr"
    def __init__(self, langs=['en']):
        # use cpu on mac M4; quantize False can avoid some issues
        self.reader = easyocr.Reader(langs, gpu=False, quantize=False)
    def run_image(self, img_path):
        res = self.reader.readtext(img_path)
        texts = [r[1] for r in res]
        return "\n".join(texts)
