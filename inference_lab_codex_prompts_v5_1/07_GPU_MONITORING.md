# Prompt 07 — Télémétrie GPU et environnement sans dépendance intrusive

Appliquer le contrat. Implémenter src/inference_lab/telemetry/gpu.py et
`python -m inference_lab.telemetry.gpu`, avec collect_hardware_metadata().
Le chemin CPU n'exige pas torch/vLLM/CUDA. Sur Kaggle, utiliser outils NVIDIA/NVML
accessibles ; ne pas exiger DCGM ou Docker. DCGM Exporter reste optionnel sur hôte dédié.

Capturer nvidia-smi et nvidia-smi topo -m, GPU UUID/index, VRAM réellement observée,
compute capability, GPU présents/visibles/utilisés, TP et topologie. Distinguer CUDA
supportée par pilote, runtime PyTorch et toolkit local s'il existe. Ne pas coder
16 GB comme valeur mesurée ni tous les GPU visibles comme GPU utilisés.

Pendant un run, exporter raw/gpu.jsonl horodaté avec cadence explicitement configurée :
utilisation/mémoire/température/puissance disponibles. Compteur non disponible = null
et raison ; ne pas imiter SM activity avec un autre compteur. Collecte limitée au
strict nécessaire, sans kernel/profiler concurrent au benchmark serving.
Pas d'hypothèse que deux GPU doublent le débit ni que leur mémoire est unifiée.

Tests CPU mocks : absence de GPU/packages, sorties NVIDIA partielles, deux GPU
présents dont un utilisé, compteur non supporté et erreur de collecte non masquée.
