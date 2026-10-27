# Priorisation v5.1

Pour la version portfolio, trois recommandations sont **fortement recommandées** sans devenir des Must du MVP :

- S1 — budget mémoire explicite ;
- S3 — RMSNorm comme vraie étude de performance ;
- S4 — analyse causale des expériences.

Elles doivent être intégrées une fois le MVP de mesure stable.

---

# InferenceLab — Should improve

**Référence :** PDD v1.2 / pack Codex v5 — **Date :** 17 octobre 2026  
**Statut :** recommandations non bloquantes, non activées par défaut.

Ces propositions approfondissent les expériences déjà prévues. Elles ne constituent
ni un backlog obligatoire ni l'autorisation de créer de nouvelles fonctionnalités.
Les Must fix sont déjà intégrés dans le PDD, le contrat et les prompts ; les points
ci-dessous décrivent l'amélioration restante au-delà de cette base.

## Vue de décision

| ID | Amélioration | Moment utile | Livrable limité |
| --- | --- | --- | --- |
| S1 | Budget mémoire explicite | Avant campagnes contexte/concurrence | Calcul et confrontation aux allocations observées |
| S2 | Interprétation des compteurs GPU | Avant rapport AWQ/saturation | Dictionnaire et limites des mesures |
| S3 | Analyse RMSNorm approfondie | Pendant Kernel Lab | Protocole et profil expliqués |
| S4 | Analyse causale du serving | Pendant campagnes système | Une explication étayée par expérience |
| S5 | Réduction de l'effort périphérique | Avant extension après MVP | Décisions de scope explicites |
| S6 | Portée Kaggle et quantification | Avant publication | Limites et provenance documentées |

## S1 — Expliciter le budget mémoire

**Problème.** « Le modèle tient sur une T4 » ne décrit pas la place disponible pour le
KV cache, les activations, le workspace, les graphs et le contexte CUDA. Une concurrence
client de 32 ne signifie pas que 32 séquences longues tiennent simultanément en mémoire.

**Impact.** OOM, préemptions ou chute de débit peuvent être mal attribués au scheduler.
Les essais deviennent coûteux si les tailles sont choisies sans ordre de grandeur.

**Modification concrète.** Pour le checkpoint retenu, ajouter au rapport existant :

1. taille approximative des poids et limites de cette estimation ;
2. coût KV théorique par token, puis budget pour les longueurs/concurrences étudiées ;
3. capacité KV rapportée par vLLM et allocations observées ;
4. différence entre ce budget théorique et le comportement réellement mesuré.

Pour un transformer dense classique en single GPU :

\[
M_{KV/token}=2\times L\times H_{KV}\times D_{head}\times octets.
\]

Exemple de calcul, pas résultat Kaggle : Qwen2.5-3B possède 36 couches, 2 têtes KV
et une dimension de tête 128. En FP16 : 36 KiB/token, soit environ 9 GiB pour
32 séquences de 8 192 tokens de KV. Ajouter ensuite poids et autres allocations.
Pour TP, tenir compte du sharding ou de la réplication effective des têtes KV ;
ne pas diviser automatiquement toute la mémoire par le nombre de GPU.

**Critère de satisfaction.** Une estimation explique les tailles choisies et ses
écarts avec les observations, sans remplacer les mesures.

**Prompts concernés :** 03, 07B, 09, 17. Pas de nouveau service ni simulateur requis.

## S2 — Interpréter correctement VRAM et utilisation GPU

**Problème.** La mémoire globale utilisée ne distingue pas poids, réserve KV et
occupation KV. AWQ peut libérer de la mémoire immédiatement réutilisée par le runtime.
GPU utilization n'est ni occupancy, ni SM activity, ni utilisation Tensor Cores.

**Impact.** Risque de conclure qu'AWQ n'économise pas de mémoire, ou qu'un GPU à
100 % est nécessairement limité par le calcul.

**Modification concrète.** Dans les résultats/dashboards existants :

- distinguer mémoire globale observée, poids si connus et capacité/usage KV ;
- expliciter source, unité, cadence et sémantique de chaque compteur ;
- qualifier un maximum par polling comme pic échantillonné ;
- ne pas remplacer un compteur matériel indisponible par une métrique différente ;
- expliquer que mémoire réservée et mémoire activement utilisée ne sont pas équivalentes.

Le contrat v5 exige déjà sources, cadence, GPU utilisés et valeurs absentes. Cette
recommandation porte sur l'analyse et la présentation approfondies des mesures.

**Critère de satisfaction.** Le rapport AWQ distingue économie des poids, réaffectation
au KV cache et capacité du serving, même si la VRAM totale observée varie peu.

**Prompts concernés :** 07, 09, 12, 17. Pas d'obligation de DCGM/Nsight sur Kaggle.

## S3 — Transformer RMSNorm en étude GPU défendable

**Problème.** Un tableau eager/compile/Triton peut surtout mesurer fusion et nombre
de lancements. Un speedup isolé n'explique pas le comportement matériel.

**Impact.** Signal insuffisant pour un poste GPU Engineer malgré une démonstration correcte.

**Modification concrète.** Approfondir le Kernel Lab existant :

1. définir précisément RMSNorm : epsilon, poids, accumulation FP32, casts, strides ;
2. utiliser une référence numérique indépendante et justifier les tolérances ;
3. inclure les hidden sizes du modèle et des nombres de lignes correspondant aux
   régimes decode et prefill ;
4. vérifier que torch.compile exécute effectivement le chemin attendu, sans fallback
   eager silencieux ; exclure compilation/autotuning des timings ;
5. expliciter chronométrage CUDA, répétitions, politique de cache et overhead de lancement ;
6. définir les octets utilisés pour l'effective bandwidth ; une estimation logique
   n'est pas une mesure directe du trafic DRAM ;
7. inclure un profil disponible expliquant un cas rapide et/ou lent.

**Critère de satisfaction.** Au moins une conclusion relie shape, accès mémoire ou
lancements au résultat, avec limites. Aucun speedup minimal n'est imposé.

**Prompts concernés :** 15, 17. Précondition : stack T4 qualifiée par 00B.

## S4 — Renforcer l'analyse causale des expériences système

**Problème.** Des courbes de TTFT/débit et des compteurs GPU ne départagent pas à elles
seules saturation mémoire, queueing, préemptions, scheduling et overhead client.

**Impact.** Rapport descriptif qui ne démontre pas suffisamment une démarche d'optimisation.

**Modification concrète.** Pour chaque expérience déjà prévue, rédiger quatre éléments :

- hypothèse préalable ;
- observation avec dispersion/coverage ;
- explication soutenue par les données ;
- explications alternatives non éliminées et limites.

Pour le scheduler, choisir une variable interprétable (`max_num_batched_tokens` ou
`max_num_seqs`) et fixer les autres. Ne pas traiter `max_model_len` comme un réglage
de scheduling isolé. Mettre en relation attente, occupation KV et latence dans les
mesures déjà accessibles, sans prétendre instrumenter ce qui ne l'est pas.

La distinction entre écart de p95 direct/gateway et overhead individuel est déjà
obligatoire dans v5. L'approfondissement consiste à expliquer les écarts observés
sans leur attribuer automatiquement une cause locale au proxy.

**Critère de satisfaction.** Une hypothèse est soutenue ou réfutée explicitement ; un
résultat négatif ou incertain est accepté comme résultat d'ingénierie.

**Prompts concernés :** 09, 12, 17. Pas d'ajout automatique d'une nouvelle campagne.

## S5 — Contenir le coût des composants périphériques

**Problème.** Watcher de modèles, workflow GPU dédié, persistance détaillée, nombreuses
ADR et hardening de production peuvent absorber le temps réservé aux mesures GPU.

**Impact.** Dépôt riche en infrastructure, mais expériences et explications trop faibles
pour le poste ciblé.

**Modification concrète.** Après le MVP, décider explicitement ce qui sert la démo :

- PostgreSQL reste un historique importé, conformément au contrat ;
- limiter les ADR aux décisions ayant un vrai compromis à expliquer ;
- utiliser les dashboards pour répondre à des questions, sans travail graphique excessif ;
- ne lancer watcher, CI GPU ou commentaires automatiques qu'après sélection d'une extension ;
- garder limites/timeouts/secrets dès le développement, mais proportionner rate limiting,
  threat model et rétention au profil réellement démontré.

**Critère de satisfaction.** Le rapport de performance et le Kernel Lab ne sont pas
repoussés pour terminer une infrastructure non nécessaire aux expériences.

**Prompts concernés :** 11, 12, 14, 16, 17. Aucun changement du scope ne doit être implicite.

## S6 — Délimiter les conclusions Kaggle et AWQ

**Problème.** Deux sessions avec le même modèle de GPU ne constituent pas un hardware
identique. Topologie, ressources hôte et état thermique peuvent varier. Une comparaison
performance AWQ ne mesure pas la qualité linguistique du modèle.

**Impact.** Résultats TP transférés abusivement à NVLink/GPU modernes ; claims qualité
ou coût GPU non soutenus par les données.

**Modification concrète.** Dans le rapport existant :

- distinguer répétabilité dans une session et reproductibilité entre sessions ;
- expliquer les limites de l'interconnexion PCIe T4 et du chemin NCCL réellement observé ;
- conserver topologie/logs/backend et comparer dans une même session en priorité ;
- préciser si téléchargement, compilation et warmup sont exclus du coût mesuré ;
- limiter les conclusions AWQ à mémoire/performance en l'absence d'évaluation de qualité ;
- indiquer contraintes de quota, réseau, disque et export effectivement rencontrées,
  sans figer une limite Kaggle supposée permanente.

**Critère de satisfaction.** Chaque claim indique son domaine de validité et ne laisse
pas croire qu'un faible error_rate technique prouve la qualité du modèle quantifié.

**Prompts concernés :** 07B, 09, 17. Pas de suite d'évaluation qualité ajoutée par défaut.

## Références techniques

Sources officielles consultées lors de la revue du 17 octobre 2026 ; elles évoluent.
Revalider la documentation correspondant aux versions effectivement qualifiées.

- [Configuration Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)
- [Métriques vLLM](https://docs.vllm.ai/en/stable/usage/metrics/)
- [Spécifications NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [Documentation Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)

## Sélection avant implémentation

Pour appliquer une recommandation, noter son ID, les prompts concernés et le critère
retenu dans une issue ou une décision du dépôt. Ne pas lancer automatiquement les six
améliorations ni créer un service supplémentaire pour les satisfaire.
