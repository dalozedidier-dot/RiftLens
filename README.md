# RiftLens

Flux nu, brut, sans ajout ni habillage. **Tout** est observé tel quel, sans narration artificielle.

Prototype v0.1 du module RiftLens.

## Installation

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## Lancement

```
python src/rift_lens.py tests/data/test_multi.csv --corr-threshold 0.6 --output-dir outputs
```

### Paramètres

- `--corr-threshold` : seuil de corrélation (float).  
  Recommandation (exemple fourni) : `0.6`. Pour un comportement déterministe, passer explicitement la valeur.

## Objectif

Explorer la rupture de cohérence locale via les graphes et corrélations sans narratif.
