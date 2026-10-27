# InferenceLab — Optional / stretch

**Référence :** PDD v1.2 / pack Codex v5 — **Date :** 17 octobre 2026  
**Statut :** extensions conditionnelles, hors acceptation du MVP système.

Ce document organise les possibilités déjà identifiées ; il n'ajoute pas un backlog
obligatoire. Le Kernel Lab RMSNorm lui-même reste requis pour le portfolio complet.
Seule son intégration dans vLLM est optionnelle. Un chemin GPU non testé ne doit pas
être présenté comme disponible parce qu'il figure ici.

## Vue de décision

| ID | Extension | Préconditions | Arrêt acceptable |
| --- | --- | --- | --- |
| O1 | Intégration RMSNorm dans vLLM | Kernel correct, benchmarké, extension vérifiée | Incompatibilité proprement documentée |
| O2 | Campagnes ~7B | Campagnes ~3B valides, budget session/disque/GPU | Pas de qualification ou absence de comparaison contrôlée |
| O3 | Registry automatique et CI GPU dédiée | MVP stable et besoin concret | Exécution/import manuels suffisants |

## O1 — Intégration RMSNorm dans le chemin vLLM

**Problème.** Une intégration forcée peut nécessiter maintenance d'un fork, adaptation
à une opération fusionnée ou gestion incorrecte du stream/lifetime des tenseurs.

**Impact.** Risque de casser le serving pour une optimisation dont l'effet end-to-end
est faible ou nul. Une intégration réussie ne prouve pas un gain de performance.

**Modification concrète si sélectionnée.** Exécuter le prompt 15B seulement après :

1. qualification de la stack et correctness du kernel ;
2. identification de l'opération réellement invoquée par le checkpoint choisi,
   dont éventuelle fusion residual/RMSNorm ;
3. vérification de l'API out-of-tree/CustomOp de la version installée ;
4. démonstration d'un lancement sans copie, sur le stream approprié, avec dtype,
   strides, device, lifetime et sémantique compatibles ;
5. flag d'activation explicite et retour simple à la baseline.

Privilégier le point d'extension existant ; un kernel Triton acceptant des tenseurs
compatibles ne nécessite pas systématiquement un binding natif. Ne pas introduire
CUDA C++ ou un fork durable pour finir cette extension.

**Validation.** Correctness de l'opération, smoke de génération réelle, puis mêmes
campagnes end-to-end baseline/candidate avec changements autorisés. Conserver résultats
nuls ou négatifs et coût de maintenance.

**Critère d'arrêt.** Si l'extension propre n'existe pas ou si la sémantique diffère,
produire une note de non-intégration. Cela ne retire pas la valeur du microbenchmark.

**Livrable limité.** Note technique, prototype activable si viable, artifacts et rapport.

## O2 — Campagnes Qwen ~7B

**Problème.** Le téléchargement, le stockage et le KV cache augmentent le coût d'itération.
Un 7B FP16 sur TP2 et un 7B AWQ sur TP1 changent simultanément plusieurs facteurs.

**Impact.** Consommer le budget Kaggle ou publier un gain causal non démontré.

**Modification concrète si sélectionnée.** Garder le checkpoint configurable et qualifier
les chemins séparément. Réutiliser les workloads/artifacts existants, sans étendre la
plateforme ou multiplier toute la matrice d'expériences.

- Si les deux conditions tiennent à ressources identiques : étude AWQ contrôlée possible.
- Si le même checkpoint/dtype fonctionne à TP1 et TP2 : étude TP contrôlée possible.
- Sinon : comparaison explicite de configurations de déploiement, avec ressources,
  capacité et différences toutes déclarées. Ne pas calculer un speedup TP à partir de
  modèles ou de précisions différents.

**Validation.** Provenance des checkpoints, versions, hardware utilisé, configuration
effective, conformité du workload, répétitions et artifacts. Aucun seuil de débit promis.

**Critère d'arrêt.** Manque de temps session/espace ou impossibilité de qualifier le
chemin ; les études ~3B restent le parcours principal.

**Livrable limité.** Une campagne ciblée et son rapport, sans nouveau backend.

## O3 — Registry automatique et CI GPU dédiée

**Problème.** Automatiser la découverte d'une révision et les exécutions GPU ajoute
de l'exploitation avant de garantir de bonnes mesures.

**Impact.** Complexité périphérique et risque de confondre pipeline fonctionnel avec
preuve de performance reproductible.

**Modification concrète si sélectionnée.** Séparer deux choix indépendants :

| Sous-option | Travail autorisé | Limites |
| --- | --- | --- |
| O3a — Registry léger | Prompt 11 : détecter une nouvelle révision et proposer une candidate | Pas de poids téléchargés ni promotion automatiques ; résoudre un commit immuable avant exécution |
| O3b — CI GPU | Étendre 14 pour runner dédié réellement disponible, déclenchement explicite, artifacts et comparateur | Kaggle n'est pas un runner permanent ; pas d'achat de GPU ni publication de commentaire sans autorisation |

L'import manuel et la validation CPU des artifacts Kaggle restent suffisants pour
le projet initial. La CI CPU du dépôt n'est pas optionnelle.

**Validation.** Droits minimaux, versions/identités traçables, quatre verdicts et erreurs
structurées. Tout résumé automatique doit respecter les règles de comparabilité.

**Critère d'arrêt.** Absence de besoin de récurrence ou de runner adapté. Ne pas créer
une infrastructure cloud pour cocher cette option.

**Livrable limité.** Une seule sous-option sélectionnée à la fois ; procédure documentée.

## Réserves existantes, sans activation

Ces éléments restent différés et ne sont pas requis par ce pack :

- **Charge open-loop** : à sélectionner uniquement si l'on veut étudier un taux
  d'arrivée imposé. Définir alors temps planifiés/réels, retard client, file et rejets ;
  ne pas interpréter rétroactivement les campagnes fermées comme de l'open-loop.
- **OpenTelemetry** : conserver comme future work. Logs/métriques/artifacts suffisent
  au MVP ; ne pas ajouter de stack de traces par défaut.

Les contextes 4k/8k et concurrence 64 restent des paramètres exploratoires conditionnels
au budget mémoire et à la qualification, pas de nouvelles obligations.

## Hors scope maintenu

Pas de Kubernetes, multi-node, nouveau runtime, GEMM custom, remplacement de
FlashAttention, frontend custom ni réécriture vLLM. Une option ne constitue pas une
autorisation implicite d'élargir ces limites.

## Décision et compte rendu

Avant une option, consigner ID, question expérimentale, préconditions vérifiées et
critère d'arrêt. Après l'essai, conserver résultat réel et limites. « Non viable dans
ce cadre » est un résultat acceptable ; aucun gain positif n'est exigé.
