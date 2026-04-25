"""Create comprehensive detector comparison notebook for thesis."""
import json
from pathlib import Path

def md(src): return {'cell_type':'markdown','metadata':{},'source':src}
def code(src): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':src}

cells = []

cells.append(md([
    "# Experimente Detector — Comparatie Completa A22 / A3 / A3-retrain / A4\n",
    "\n",
    "Acest notebook prezinta evolutia detectorului de gunoi de-a lungul tuturor experimentelor.\n",
    "\n",
    "| Experiment | Model | Dataset | Imagini | Epochs | **mAP50** | mAP50-95 | Precision | Recall |\n",
    "|------------|-------|---------|---------|--------|-----------|---------|-----------|--------|\n",
    "| A22 | YOLOv8n | Parks | 1544 | 150 | 0.393 | 0.281 | 0.707 | 0.286 |\n",
    "| A3-final | YOLOv8s | Parks | 1544 | 150 | 0.443 | 0.321 | 0.623 | 0.406 |\n",
    "| A3-retrain | YOLOv8s | Parks (resized) | 1544 | 150 | **0.482** | 0.356 | 0.628 | 0.437 |\n",
    "| **A4** | YOLOv8s | Parks+TACO | **2307** | 50 | **0.666** | **0.523** | **0.857** | **0.552** |\n",
]))

cells.append(code([
    "from pathlib import Path\n",
    "import json\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "from IPython.display import Image, display\n",
    "\n",
    "REPO = Path().resolve().parents[1]\n",
    "A3R_DIR = REPO / 'runs' / 'detect' / 'parks-trash-A3-retrain'\n",
    "A4_DIR  = REPO / 'runs' / 'detect' / 'parks-trash-A4'\n",
    "\n",
    "print('A3-retrain weights:', (A3R_DIR / 'weights' / 'best.pt').exists())\n",
    "print('A4 weights:        ', (A4_DIR / 'weights' / 'best.pt').exists())\n",
]))

cells.append(md(["## 1. Tabel comparativ complet\n"]))

cells.append(code([
    "data = {\n",
    "    'Experiment': ['A22 (baseline)', 'A3-final (original)', 'A3-retrain', 'A4 (Parks+TACO)'],\n",
    "    'Model':      ['YOLOv8n', 'YOLOv8s', 'YOLOv8s', 'YOLOv8s'],\n",
    "    'Dataset':    ['Parks', 'Parks', 'Parks resized', 'Parks+TACO'],\n",
    "    'Imagini':    [1544, 1544, 1544, 2307],\n",
    "    'Epochs':     [150, 150, 150, 50],\n",
    "    'Precision':  [0.707, 0.623, 0.628, 0.857],\n",
    "    'Recall':     [0.286, 0.406, 0.437, 0.552],\n",
    "    'mAP50':      [0.393, 0.443, 0.482, 0.666],\n",
    "    'mAP50-95':   [0.281, 0.321, 0.356, 0.523],\n",
    "}\n",
    "df = pd.DataFrame(data).set_index('Experiment')\n",
    "display(df.style\n",
    "    .highlight_max(subset=['Precision','Recall','mAP50','mAP50-95'], color='#90EE90')\n",
    "    .format({'Precision':'{:.3f}','Recall':'{:.3f}','mAP50':'{:.3f}','mAP50-95':'{:.3f}'})\n",
    ")\n",
]))

cells.append(md(["## 2. Evolutia mAP50 — toti detectorii\n"]))

cells.append(code([
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Absolute mAP50 bar chart\n",
    "experiments = ['A22', 'A3-final', 'A3-retrain', 'A4']\n",
    "map50_vals  = [0.393, 0.443, 0.482, 0.666]\n",
    "colors = ['#d4e6f1', '#85c1e9', '#2e86c1', '#1a5276']\n",
    "bars = axes[0].bar(experiments, map50_vals, color=colors, edgecolor='white', linewidth=1.5)\n",
    "axes[0].set_ylabel('mAP50')\n",
    "axes[0].set_title('mAP50 per experiment')\n",
    "axes[0].set_ylim(0, 0.75)\n",
    "axes[0].axhline(0.443, color='gray', linestyle='--', alpha=0.5, label='A3-final baseline')\n",
    "axes[0].legend()\n",
    "for bar, val in zip(bars, map50_vals):\n",
    "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n",
    "                 f'{val:.3f}', ha='center', fontweight='bold')\n",
    "axes[0].grid(axis='y', alpha=0.3)\n",
    "\n",
    "# Training curves overlay (A3-retrain vs A4)\n",
    "import pandas as pd\n",
    "for exp_dir, label, color in [\n",
    "    (A3R_DIR, 'A3-retrain (parks, 150ep)', '#2e86c1'),\n",
    "    (A4_DIR,  'A4 (parks+TACO, 50ep)',     '#1a5276'),\n",
    "]:\n",
    "    csv = exp_dir / 'results.csv'\n",
    "    if csv.exists():\n",
    "        df_r = pd.read_csv(csv)\n",
    "        df_r.columns = df_r.columns.str.strip()\n",
    "        axes[1].plot(df_r['epoch'], df_r['metrics/mAP50(B)'],\n",
    "                     label=label, color=color, linewidth=2)\n",
    "axes[1].axhline(0.443, color='gray', linestyle='--', alpha=0.6, label='A3-final original')\n",
    "axes[1].set_xlabel('Epoch')\n",
    "axes[1].set_ylabel('mAP50')\n",
    "axes[1].set_title('Curbe training: A3-retrain vs A4')\n",
    "axes[1].legend()\n",
    "axes[1].grid(alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "out = REPO / 'runs' / 'detector_comparison.png'\n",
    "plt.savefig(str(out), dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Salvat:', out)\n",
]))

cells.append(md(["## 3. Curbe training A3-retrain\n"]))

cells.append(code([
    "display(Image(str(A3R_DIR / 'results.png'), width=900))\n",
]))

cells.append(md(["## 4. Predictii comparative — A3-retrain vs A4\n"]))

cells.append(code([
    "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
    "imgs = [\n",
    "    (A3R_DIR / 'val_batch0_labels.jpg', 'Ground Truth'),\n",
    "    (A3R_DIR / 'val_batch0_pred.jpg',   'A3-retrain pred'),\n",
    "    (A4_DIR  / 'val_batch0_labels.jpg', 'Ground Truth'),\n",
    "    (A4_DIR  / 'val_batch0_pred.jpg',   'A4 pred'),\n",
    "]\n",
    "for ax, (p, title) in zip(axes.flat, imgs):\n",
    "    if p.exists():\n",
    "        ax.imshow(mpimg.imread(str(p)))\n",
    "    ax.set_title(title, fontsize=12)\n",
    "    ax.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

cells.append(md(["## 5. Evaluare pe test set — A3-retrain\n"]))

cells.append(code([
    "from ultralytics import YOLO\n",
    "\n",
    "model_a3 = YOLO(str(A3R_DIR / 'weights' / 'best.pt'))\n",
    "m3 = model_a3.val(\n",
    "    data=str(REPO / 'datasets/parks_detect_A3_resized/dataset.yaml'),\n",
    "    split='test', imgsz=640, batch=8, device=0, workers=0, verbose=False,\n",
    ")\n",
    "print('=== A3-RETRAIN TEST SET ===')\n",
    "print(f'mAP50:     {m3.box.map50:.4f}')\n",
    "print(f'mAP50-95:  {m3.box.map:.4f}')\n",
    "print(f'Precision: {m3.box.mp:.4f}')\n",
    "print(f'Recall:    {m3.box.mr:.4f}')\n",
]))

cells.append(md([
    "## Concluzii\n",
    "\n",
    "Evolutia clara a performantelor pe parcursul experimentelor:\n",
    "\n",
    "1. **A22 → A3**: trecerea de la YOLOv8n la YOLOv8s (+5pp mAP50) — modelul mai mare ajuta.\n",
    "2. **A3 → A3-retrain**: reproducere cu imagini rezimensionate (+3.9pp) — calitatea preprocessing-ului conteaza.\n",
    "3. **A3-retrain → A4**: adaugarea TACO (+18.4pp mAP50) — diversitatea datelor e factorul dominant.\n",
    "\n",
    "**Concluzie:** Modelul A4 (`runs/detect/parks-trash-A4/weights/best.pt`) este selectat pentru productie\n",
    "cu mAP50=**0.666**, o imbunatatire de **+22.3pp** fata de A3-final original.\n",
]))

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.12.0'}
    },
    'cells': cells
}

out = Path('d:/TrashDetectionSystem/notebooks/training/02_detector_comparison.ipynb')
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Notebook creat: {out}  ({len(cells)} celule)')
