# Zero to Skyjo

## Project layout (modular)
- `zero_to_skyjo/envs`: RL environments (same interface across envs)
- `zero_to_skyjo/agents`: RL agents (DQN now, others later)
- `zero_to_skyjo/utils`: shared utilities (replay buffer, seeding)
- `configs`: experiment configs (TOML)
- `runs`: outputs per run (metrics, plots, models, config snapshot)

## Models
- DQN (dueling, double update in training loop)
- PPO (actor-critic with training loop)
- Tabular Q-learning (compact state)
- Random baseline
- Greedy baselines (discard-first, discard>5 -> draw)

## Quick start
- One command (mode from config): `uv run main.py --config configs/mini_skyjo_dqn.toml`
- Solo Skyjo train: `uv run main.py --config configs/solo_skyjo_dqn.toml --mode train`
- Solo Skyjo PPO train: `uv run main.py --config configs/solo_skyjo_ppo.toml --mode train`
- Solo Skyjo Tabular Q train: `uv run main.py --config configs/solo_skyjo_tabular_q.toml --mode train`
- Train (explicit): `uv run main.py --config configs/mini_skyjo_dqn.toml --mode train`
- Eval: `uv run main.py --config configs/mini_skyjo_dqn.toml --mode eval --model runs/<exp>/<ts>/model_best.pt`
- Demo (terminal): `uv run main.py --config configs/mini_skyjo_dqn.toml --mode demo --model runs/<exp>/<ts>/model_best.pt`
- Eval (standalone): `uv run eval_agent.py --config configs/mini_skyjo_dqn.toml --model runs/<exp>/<ts>/model_best.pt`
- Demo UI (pygame): `uv run demo_ui.py --config configs/mini_skyjo_dqn.toml --model runs/<exp>/<ts>/model_best.pt`
- Baseline eval (random): `uv run eval_agent.py --config configs/solo_skyjo_random.toml`
- Baseline eval (greedy min): `uv run eval_agent.py --config configs/solo_skyjo_greedy_min.toml`
- Baseline eval (greedy threshold): `uv run eval_agent.py --config configs/solo_skyjo_greedy_threshold.toml`

Outputs go to `runs/<experiment>/<timestamp>/` with `model_best.pt`, `model_final.pt`, `metrics.csv`, and plots.


## Objectif du projet
L'objectif de ce projet est de développer une intelligence artificielle capable de jouer au Skyjo, un jeu de cartes stratégique, tout en minimisant son score. Ce projet est structuré en plusieurs phases pour garantir une progression méthodique et efficace.

### Règles du Skyjo
- Le jeu se joue avec un plateau de cartes numérotées.
- Chaque joueur a un tableau de cartes cachées.
- Le but est de minimiser son score en remplaçant ses cartes par des cartes de plus faible valeur.
- Les joueurs jouent à tour de rôle en tirant des cartes et en décidant de les garder ou de les défausser.

## Phase 1: Expérimentation et choix de l'architecture

### Objectif
L'objectif de cette phase est d'expérimenter différentes architectures de modèles pour identifier celle qui est la plus adaptée au problème. Pour cela, nous utilisons une version simplifiée du Skyjo, appelée "Mini Skyjo".

### Règles simplifiées du Mini Skyjo
- Le jeu utilise un deck de 36 cartes, avec des valeurs allant de 1 à 6, chaque valeur ayant 6 copies.
- Chaque joueur commence avec 6 cartes face cachée.
- À chaque tour, deux cartes sont proposées : une visible et une cachée. Le joueur doit choisir l'une des deux cartes et remplacer une carte de sa main.
- Le jeu se termine lorsque toutes les cartes du joueur sont visibles ou que le deck contient moins de 2 cartes.
- Le score final est la somme des valeurs des cartes visibles plus le nombre d'échanges effectués.

### Entraînement et test du modèle
- Pour entraîner le modèle sur le Mini Skyjo, utilisez la commande suivante :
  ```
  uv run -m zero_to_skyjo.train --config configs/mini_skyjo_dqn.toml
  ```
- Pour tester le modèle avec un affichage du jeu et des étapes dans le terminal, utilisez :
  ```
  uv run -m zero_to_skyjo.demo --config configs/mini_skyjo_dqn.toml --model runs/<exp>/<ts>/model_best.pt --delay 0.2
  ```

### Résultats
Les performances de l'architecture du modèle sur le Mini Skyjo sont bonnes, ce qui valide l'approche choisie pour cette phase. Les métriques et courbes d'apprentissage montrent une amélioration significative au fil des épisodes d'entraînement.

![Courbe d'apprentissage](learning_curve.png)

## Phase 2: Intégration avec l'API

### Objectif
La deuxième phase consiste à connecter tous les éléments développés. Cela inclut l'intégration de l'API, développée en Rust par Romain BOURDAIN, avec le modèle d'intelligence artificielle. Cette étape permettra de passer d'une version simplifiée à une version complète et interactive du Skyjo, où l'IA pourra interagir avec une API robuste et performante.
