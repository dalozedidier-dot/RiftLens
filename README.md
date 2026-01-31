# RiftLens

Prototype v0.1 du module RiftLens.

## Installation
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install

## Lancement
python src/rift_lens.py tests/data/test_multi.csv --corr-threshold 0.6 --output-dir outputs

## Objectif
Construire un graphe de cohérence descriptif entre variables et produire un rapport auditable.
## Paramètres
- `--corr-threshold` : seuil de corrélation (valeur absolue). Défaut: `0.6`.
- `--output-dir` : dossier de sortie (rapports + artefacts).
