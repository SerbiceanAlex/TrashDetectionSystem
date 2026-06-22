"""
Clasificarea materialului pentru un decupaj de obiect (etapa a 2-a a pipeline-ului).

Detectorul YOLO produce casetele de deșeu (în backend), iar fiecare decupaj e
clasificat aici pe material cu clasificatorul YOLOv8-cls.
"""


def classify_crop(classifier, crop, imgsz, class_names):
    """Clasifică materialul unui decupaj; întoarce (nume_material, încredere)."""
    if classifier is None:
        return "unknown", 0.0
    result = classifier.predict(crop, imgsz=imgsz, verbose=False)[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        return "unknown", 0.0

    top_index = int(probs.top1)
    top_conf = float(probs.top1conf.item() if hasattr(probs.top1conf, "item") else probs.top1conf)
    return class_names.get(top_index, str(top_index)), top_conf
