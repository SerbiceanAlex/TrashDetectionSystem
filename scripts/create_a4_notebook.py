"""Create the A4 training notebook for thesis."""
import json
from pathlib import Path

def md(src): return {'cell_type':'markdown','metadata':{},'source':src}
def code(src): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':src}

cells = []

cells.append(md([
    "# Experiment A4 — Detector extins: Parks + TACO\n",
    "\n",
    "**Obiectiv:** Extinderea dataset-ului de antrenare cu imagini TACO (Trash Annotations in Context)\n",
    "pentru a imbunatati acoperirea diverselor tipuri de deseuri, inclusiv obiecte mici.\n",
    "\n",
    "| | A3-final (baseline) | **A4 (acest experiment)** |\n",
    "|---|---|---|\n",
    "| Dataset | Parks only — 1544 img | Parks + TACO — **2307 img** |\n",
    "| Model base | YOLOv8s pretrained | YOLOv8s pretrained |\n",
    "| Epochs | 50 | 50 |\n",
    "| imgsz | 640 | 640 |\n",
    "| **mAP50** | 0.443 | **0.666** |\n",
    "| **mAP50-95** | 0.321 | **0.523** |\n",
    "| Precision | 0.623 | **0.857** |\n",
    "| Recall | 0.406 | **0.552** |\n",
]))

cells.append(code([
    "import sys\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "from IPython.display import Image, display\n",
    "\n",
    "REPO = Path().resolve().parents[1]\n",
    "A4_DIR = REPO / 'runs' / 'detect' / 'parks-trash-A4'\n",
    "A4_WEIGHTS = A4_DIR / 'weights' / 'best.pt'\n",
    "\n",
    "print('A4 weights:', A4_WEIGHTS.exists(), f'({A4_WEIGHTS.stat().st_size//1024//1024}MB)')\n",
    "print('Results dir:', A4_DIR.exists())\n",
]))

cells.append(md(["## 1. Curbe de antrenare\n"]))

cells.append(code([
    "display(Image(str(A4_DIR / 'results.png'), width=900))\n",
]))

cells.append(md(["## 2. Curbe Precision-Recall si F1\n"]))

cells.append(code([
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "axes[0].imshow(mpimg.imread(str(A4_DIR / 'BoxPR_curve.png')))\n",
    "axes[0].axis('off')\n",
    "axes[0].set_title('PR Curve')\n",
    "axes[1].imshow(mpimg.imread(str(A4_DIR / 'BoxF1_curve.png')))\n",
    "axes[1].axis('off')\n",
    "axes[1].set_title('F1 Curve')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

cells.append(md(["## 3. Confusion Matrix\n"]))

cells.append(code([
    "display(Image(str(A4_DIR / 'confusion_matrix_normalized.png'), width=600))\n",
]))

cells.append(md(["## 4. Predictii pe setul de validare\n"]))

cells.append(code([
    "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
    "axes[0].imshow(mpimg.imread(str(A4_DIR / 'val_batch0_labels.jpg')))\n",
    "axes[0].axis('off')\n",
    "axes[0].set_title('Ground Truth')\n",
    "axes[1].imshow(mpimg.imread(str(A4_DIR / 'val_batch0_pred.jpg')))\n",
    "axes[1].axis('off')\n",
    "axes[1].set_title('Predictii A4')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

cells.append(md(["## 5. Evolutia mAP50 — A4 vs A3 baseline\n"]))

cells.append(code([
    "df = pd.read_csv(A4_DIR / 'results.csv')\n",
    "df.columns = df.columns.str.strip()\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "ax1.plot(df['epoch'], df['metrics/mAP50(B)'], label='A4 mAP50', color='blue', linewidth=2)\n",
    "ax1.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='A4 mAP50-95', color='orange', linewidth=2)\n",
    "ax1.axhline(0.443, color='blue', linestyle='--', alpha=0.6, label='A3 mAP50=0.443')\n",
    "ax1.axhline(0.321, color='orange', linestyle='--', alpha=0.6, label='A3 mAP50-95=0.321')\n",
    "ax1.set_xlabel('Epoch')\n",
    "ax1.set_ylabel('mAP')\n",
    "ax1.set_title('mAP A4 vs A3 baseline (linie intrerupta)')\n",
    "ax1.legend()\n",
    "ax1.grid(alpha=0.3)\n",
    "\n",
    "ax2.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='green', linewidth=2)\n",
    "ax2.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='red', linewidth=2)\n",
    "ax2.axhline(0.623, color='green', linestyle='--', alpha=0.6, label='A3 Precision=0.623')\n",
    "ax2.axhline(0.406, color='red', linestyle='--', alpha=0.6, label='A3 Recall=0.406')\n",
    "ax2.set_xlabel('Epoch')\n",
    "ax2.set_ylabel('Score')\n",
    "ax2.set_title('Precision / Recall A4 vs A3')\n",
    "ax2.legend()\n",
    "ax2.grid(alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(str(A4_DIR / 'A4_vs_A3_comparison.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Grafic salvat pentru teza:', A4_DIR / 'A4_vs_A3_comparison.png')\n",
]))

cells.append(md(["## 6. Evaluare finala pe test set\n"]))

cells.append(code([
    "from ultralytics import YOLO\n",
    "\n",
    "model = YOLO(str(A4_WEIGHTS))\n",
    "metrics = model.val(\n",
    "    data=str(REPO / 'datasets/parks_detect_A4/dataset.yaml'),\n",
    "    split='test',\n",
    "    imgsz=640,\n",
    "    batch=8,\n",
    "    device=0,\n",
    "    workers=0,\n",
    "    verbose=False,\n",
    ")\n",
    "\n",
    "print('=== REZULTATE TEST SET A4 ===')\n",
    "print(f'Precision:  {metrics.box.mp:.4f}')\n",
    "print(f'Recall:     {metrics.box.mr:.4f}')\n",
    "print(f'mAP50:      {metrics.box.map50:.4f}')\n",
    "print(f'mAP50-95:   {metrics.box.map:.4f}')\n",
]))

cells.append(md(["## 7. Tabel comparativ final — pentru teza\n"]))

cells.append(code([
    "data = {\n",
    "    'Experiment': ['A3-final (baseline)', 'A4 (Parks+TACO)'],\n",
    "    'Dataset': ['Parks 1094 img', 'Parks+TACO 1451 img'],\n",
    "    'Total imagini': [1544, 2307],\n",
    "    'Precision': [0.623, 0.857],\n",
    "    'Recall': [0.406, 0.552],\n",
    "    'mAP50': [0.443, 0.666],\n",
    "    'mAP50-95': [0.321, 0.523],\n",
    "    'Delta mAP50': ['-', '+22.3pp'],\n",
    "}\n",
    "df_cmp = pd.DataFrame(data).set_index('Experiment')\n",
    "display(df_cmp.style.highlight_max(\n",
    "    subset=['Precision', 'Recall', 'mAP50', 'mAP50-95'],\n",
    "    color='lightgreen'\n",
    "))\n",
]))

cells.append(md([
    "## Concluzii\n",
    "\n",
    "Extinderea dataset-ului cu TACO (1500 imagini, 60 categorii de deseuri reale) a produs\n",
    "imbunatatiri semnificative fata de A3-final:\n",
    "\n",
    "- **mAP50: +22.3pp** (0.443 -> 0.666)\n",
    "- **Precision: +23.4pp** (0.623 -> 0.857) — mai putine false pozitive\n",
    "- **Recall: +14.6pp** (0.406 -> 0.552) — mai putine obiecte ratate\n",
    "\n",
    "Modelul A4 (`runs/detect/parks-trash-A4/weights/best.pt`) este candidatul pentru productie.\n",
]))

nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.0'}
    },
    'cells': cells
}

out = Path('d:/TrashDetectionSystem/notebooks/training/01b_train_detector_A4.ipynb')
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Notebook creat: {out}')
print(f'Celule: {len(cells)}')
