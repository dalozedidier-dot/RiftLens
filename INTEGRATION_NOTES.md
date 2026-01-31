# Notes d'intégration

Ajouts recommandés
- Ajouter le contenu de requirements_dd_graph.txt à votre requirements.txt ou requirements.in
- Copier dd_graph_tool/ et scripts/run_dd_graph.py dans le repo cible
- Copier tests/ si vous voulez le smoke test
- Copier .github/workflows/dd_graph_isolated.yml si vous voulez le job CI isolé

Contraintes respectées
- descriptif uniquement
- pas de causalité
- pas de prédiction
- sortie auditables (PNG, JSON, GEXF)
