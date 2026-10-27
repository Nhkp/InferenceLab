# START HERE — InferenceLab v5.1

Commence par lire `PROJECT_DESIGN_DOCUMENT.md` (ou le PDD v1.2 présent dans l'archive), `BENCHMARK_CONTRACT.md`, puis `00_GLOBAL_RULES.md`.

Ensuite, **ne commence pas par construire toute la plateforme**. La première étape réelle est :

```text
00B_T4_COMPATIBILITY_QUALIFICATION.md
```

## Ordre immédiat

1. Qualifier Kaggle + T4.
2. Figer les versions réellement compatibles.
3. Valider le streaming vLLM sur 1×T4 en FP16.
4. Donner un statut explicite à `torch.compile`, Triton, TP=2 et AWQ.
5. Conserver les artifacts de qualification.
6. Ensuite seulement lancer les prompts d'implémentation.

## Jalons

- **A — Qualification GPU**
- **B — MVP système**
- **C — Portfolio avec Kernel Lab**
- **D — Stretch goals**

Le serving de référence reste sur T4. Si Triton ne peut raisonnablement pas fonctionner sur T4, l'échec doit être documenté et le Kernel Lab pourra être exécuté sur un GPU compatible séparément identifié, sans mélanger ces résultats avec les benchmarks serving T4.
