# Prompt 16 — Hardening production

Audit robustesse et sécurité dans le scope réel du portfolio.
Les limites, timeouts global/inactivité, redaction et gestion de déconnexion sont
déjà requis dans les étapes concernées : ne pas les repousser à cet audit.
Appliquer le contrat streaming et l'absence de retry dans les benchmarks. Tout
rate limiting actif et paramètre modifiant la charge doit figurer dans le manifeste.

API : limite body, limite max_tokens, timeout global, validation stricte, erreurs sans stacktrace.

Rate limiting simple configurable ou externalisation explicitement documentée.

Vérifier secrets, `.env`, redaction Authorization, prompts non loggés. Ajouter tests de non-fuite.

Database : pool, timeouts, index, rétention. Docker : non-root si possible, healthchecks, versions raisonnables, image minimale. Ajouter scan de dépendances CI sans service payant.

Créer `docs/threat_model.md` : actifs, surfaces d'attaque, abus endpoint, fuite prompts, supply-chain model, trust_remote_code, DoS contexte long, dépendances, accès GPU ; classer impact/probabilité et mitigations.
