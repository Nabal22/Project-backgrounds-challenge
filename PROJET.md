# Backgrounds Challenge — Documentation du projet

## Vue d'ensemble

Ce projet reproduit et améliore les expériences du papier **"Noise or Signal: The Role of Image Backgrounds in Object Recognition"** (Xiao et al., 2020). L'objectif est de mesurer à quel point les modèles de vision par ordinateur s'appuient sur le fond d'une image pour classifier les objets, puis de réduire cette dépendance via des techniques d'augmentation de données.

La métrique principale est le **BG-Gap** : différence d'accuracy entre `mixed_same` (fonds corrélés avec la classe) et `mixed_rand` (fonds aléatoires). Un BG-Gap élevé signifie que le modèle triche en utilisant le fond.

---

## Structure des fichiers

```
Project-backgrounds-challenge/
├── baseline.ipynb        ← Évaluation de la baseline
├── train.ipynb           ← Fine-tuning avec Mixup
├── build_in9.py          ← Construction du dataset IN-9
├── challenge_eval.py     ← Évaluation adversariale (CLI)
├── in9_eval.py           ← Évaluation par variation (CLI)
├── in_to_in9.json        ← Mapping ImageNet (1000 classes) → IN-9 (9 classes)
├── in9_classes.txt       ← Noms des 9 super-classes
├── in9/                  ← Dataset d'entraînement (~34k images)
├── bg_challenge/         ← Données de test (9 variations)
├── imagenet-mini/        ← Source ImageNet utilisée pour construire in9/
├── imagenet_models/      ← Checkpoints de modèles
├── tools/                ← Utilitaires partagés (datasets, modèles)
│   ├── datasets.py
│   ├── model_utils.py
│   └── folder.py
└── assets/               ← Images pour le README
```

---

## Fichiers principaux

### `baseline.ipynb` — Évaluation de la baseline ResNet-50

Notebook Google Colab qui évalue un ResNet-50 **prétrained sur ImageNet** (sans fine-tuning) sur les 7 variations de test IN-9.

**Ce qu'il fait :**
1. Télécharge les données de test `bg_challenge/` depuis le GitHub release
2. Télécharge le mapping `in_to_in9.json`
3. Charge ResNet-50 prétrained (1000 classes ImageNet)
4. Évalue sur chaque variation via une fonction `evaluate()` qui mappe les prédictions 1000→9 classes
5. Affiche les résultats et calcule le BG-Gap

**Résultats obtenus :**
| Variation   | Accuracy |
|-------------|----------|
| original    | 95.5%    |
| mixed_same  | 86.2%    |
| mixed_rand  | 78.9%    |
| only_fg     | 86.6%    |
| no_fg       | 46.2%    |
| only_bg_t   | 15.9%    |
| only_bg_b   | 11.6%    |
| **BG-Gap**  | **7.4%** |

---

### `train.ipynb` — Fine-tuning ResNet-50 avec Mixup

Notebook Google Colab qui **fine-tune ResNet-50** sur IN-9 avec l'augmentation **Mixup** pour réduire le BG-Gap.

**Ce qu'il fait :**
1. Monte Google Drive (source des données `in9/`)
2. Télécharge les données de test `bg_challenge/`
3. Remplace la tête de classification ResNet-50 par `Linear(2048, 9)` (9 classes directement)
4. Entraîne 10 epochs avec SGD + CosineAnnealingLR + **Mixup (α=0.2)**
5. Évalue sur les 7 variations — sans mapping (le modèle sort directement 9 logits)
6. Sauvegarde le checkpoint sur Drive
7. Génère un bar chart comparatif baseline vs Mixup

**Différence clé avec baseline.ipynb :** pas besoin de `in_to_in9.json` pour l'évaluation, le modèle est directement entraîné sur 9 classes.

**Objectif :** BG-Gap < 6% (vs 7.4% baseline).

---

### `build_in9.py` — Construction du dataset IN-9

Script Python à exécuter localement pour construire le dataset d'entraînement `in9/` à partir d'`imagenet-mini/`.

**Ce qu'il fait :**
- Lit `imagenet-mini/train/` et `imagenet-mini/val/` (structure synsets ImageNet)
- Utilise `in_to_in9.json` pour mapper chaque synset vers l'une des 9 super-classes
- Copie les images dans `in9/train/{0-8}/` et `in9/val/{0-8}/`

**Usage :**
```bash
python build_in9.py
```

**À exécuter une seule fois** avant d'uploader `in9/` sur Google Drive pour l'entraînement.

---

### `in9_eval.py` — Évaluation CLI sur une variation IN-9

Script en ligne de commande pour évaluer un checkpoint sur une variation spécifique de IN-9. Utilise les utilitaires `tools/`.

**Usage :**
```bash
# Modèle ImageNet 1000 classes
python in9_eval.py --eval-dataset mixed_rand --data-path bg_challenge/ --checkpoint /path/to/model.pt

# Modèle IN-9 9 classes
python in9_eval.py --eval-dataset mixed_rand --data-path bg_challenge/ --checkpoint /path/to/model.pt --in9
```

**Variations disponibles :** `original`, `mixed_same`, `mixed_rand`, `only_fg`, `no_fg`, `only_bg_t`, `only_bg_b`

---

### `challenge_eval.py` — Évaluation adversariale CLI

Script pour l'évaluation du **challenge officiel** : mesure le pourcentage de foregrounds "vulnérables" à des fonds adversariaux. Plus exigeant que l'évaluation sur les variations standard.

**Usage :**
```bash
python challenge_eval.py --checkpoint /path/to/model.pt --data-path bg_challenge/
```

---

### `in_to_in9.json` — Mapping de classes

Dictionnaire JSON qui mappe chaque index de classe ImageNet (0–999) vers l'une des 9 super-classes IN-9 (0–8). Utilisé par la baseline et les scripts CLI pour convertir les prédictions 1000→9 classes.

```json
{"0": 5, "1": 3, "2": 5, ...}
```

Les 9 super-classes : `Dog(0)`, `Bird(1)`, `Vehicle(2)`, `Reptile(3)`, `Carnivore(4)`, `Insect(5)`, `Instrument(6)`, `Primate(7)`, `Fish(8)`

---

### `tools/` — Utilitaires partagés

Utilisés par `in9_eval.py` et `challenge_eval.py` (pas par les notebooks).

| Fichier | Rôle |
|---------|------|
| `datasets.py` | Classes `ImageNet` et `ImageNet9` pour charger les données |
| `model_utils.py` | `make_and_restore_model()`, `eval_model()`, `adv_bgs_eval_model()` |
| `folder.py` | Variante de `ImageFolder` PyTorch avec support masques |

---

## Données

### `bg_challenge/` — Données de test

9 variations générées synthétiquement depuis ImageNet-9 :

| Dossier | Description |
|---------|-------------|
| `original/` | Images ImageNet-9 originales |
| `mixed_same/` | Foreground + fond de la même classe |
| `mixed_rand/` | Foreground + fond d'une classe aléatoire |
| `mixed_next/` | Foreground + fond de la classe suivante |
| `only_fg/` | Foreground seul (fond blanc) |
| `no_fg/` | Fond seul (foreground masqué) |
| `only_bg_t/` | Fond seul (texture) |
| `only_bg_b/` | Fond seul (flou) |
| `fg_mask/` | Masques binaires foreground (fichiers `.npy`) |

### `in9/` — Données d'entraînement

Dataset construit par `build_in9.py` depuis `imagenet-mini/`. Structure :
```
in9/
├── train/
│   ├── 0/   ← Dog
│   ├── 1/   ← Bird
│   └── ...
└── val/
    └── ...
```

---

## Workflow complet

```
1. [Local] python build_in9.py         → Crée in9/ depuis imagenet-mini/
2. [Local] Zipper in9/ et uploader sur Google Drive
3. [Colab] baseline.ipynb              → Évalue ResNet-50 baseline (BG-Gap: 7.4%)
4. [Colab] train.ipynb                 → Fine-tune avec Mixup (objectif BG-Gap < 6%)
5. Comparer les résultats dans la cellule 12 de train.ipynb
```
