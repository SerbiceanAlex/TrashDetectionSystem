"""
Clasificarea materialului pentru un decupaj de obiect (etapa a 2-a a pipeline-ului).

Detectorul YOLO produce casetele de deșeu (în backend), iar fiecare decupaj e
clasificat aici pe material cu clasificatorul YOLOv8-cls.
"""


def classify_crop(classifier, crop, imgsz, class_names, min_conf: float = 0.55):
    """Clasifică materialul unui decupaj; întoarce (nume_material, încredere).

    Clasificatorul (antrenat pe TrashNet — poze de studio) e nesigur pe crop-uri
    de webcam (out-of-distribution). Ca să nu afișeze o etichetă GREȘITĂ cu
    încredere mică (ex. „paper" pe o sticlă), sub `min_conf` întoarce „unknown" —
    o necunoaștere onestă e mai bună decât o clasificare greșită afișată ferm.
    """
    if classifier is None:
        return "unknown", 0.0
    result = classifier.predict(crop, imgsz=imgsz, verbose=False)[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        return "unknown", 0.0

    top_index = int(probs.top1)
    top_conf = float(probs.top1conf.item() if hasattr(probs.top1conf, "item") else probs.top1conf)
    if top_conf < min_conf:
        return "unknown", top_conf
    return class_names.get(top_index, str(top_index)), top_conf
