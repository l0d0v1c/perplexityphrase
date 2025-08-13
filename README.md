# PerplexityPhrase

Suite d'outils Python pour analyser la perplexité des phrases d'un texte et les trier par ordre décroissant de complexité linguistique.

## Description

Cette suite utilise le framework MLX d'Apple avec le modèle de langage SmolLM3-3B-4bit pour :

1. **Découper** un texte en phrases
2. **Calculer** la perplexité de chaque phrase (mesure de surprise/complexité pour le modèle)
3. **Stocker** les résultats dans une base SQLite pour persistance
4. **Extraire** et analyser les résultats selon différents critères
5. **Trier** les phrases par perplexité, complexité linguistique, ou autres métriques

La perplexité indique à quel point une phrase est "surprenante" ou difficile à prédire pour le modèle. Plus la perplexité est élevée, plus la phrase est complexe ou inattendue.

## Prérequis

- macOS avec puce Apple Silicon (M1/M2/M3)
- Python 3.9+
- MLX framework

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-username/perplexityphrase.git
cd perplexityphrase
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

Le premier lancement téléchargera automatiquement le modèle SmolLM3-3B-4bit (~2GB).

## Scripts disponibles

### 1. `perplexity_phrase_sorter.py` - Traitement simple
Pour analyser rapidement de petits textes.

### 2. `perplexity_batch_processor.py` - Traitement batch avec persistance
Pour traiter de très longs textes avec reprise après crash.

### 3. `extract_results.py` - Extraction et analyse des résultats
Pour analyser les résultats stockés dans la base SQLite.

## Utilisation

### Traitement simple (textes courts)

```bash
# Analyse d'un fichier texte
python perplexity_phrase_sorter.py --input exemple_texte.txt

# Analyse d'un texte direct
python perplexity_phrase_sorter.py --text "L'intelligence artificielle fascine. Les modèles évoluent rapidement."

# Mode verbose (avec détails par token)
python perplexity_phrase_sorter.py --input exemple_texte.txt --verbose
```

### Traitement batch (textes longs)

```bash
# Premier lancement - traite tout le texte long
python perplexity_batch_processor.py --input long_text_example.txt

# Après interruption - reprend où ça s'est arrêté
python perplexity_batch_processor.py --input long_text_example.txt

# Voir les statistiques de progression
python perplexity_batch_processor.py --database perplexity_cache.db --stats-only

# Voir les résultats triés
python perplexity_batch_processor.py --database perplexity_cache.db --results-only
```

### Extraction et analyse des résultats

```bash
# Toutes les phrases par perplexité décroissante
python extract_results.py perplexity_cache.db

# Top 20 phrases avec perplexité la plus élevée
python extract_results.py perplexity_cache.db --top 20

# Top 15 phrases les plus COMPLEXES (perplexité + longueur)
python extract_results.py perplexity_cache.db --complex 15

# Phrases les plus simples
python extract_results.py perplexity_cache.db --bottom 10

# Recherche de phrases contenant "intelligence"
python extract_results.py perplexity_cache.db --search "intelligence"

# Export vers fichier CSV
python extract_results.py perplexity_cache.db --output resultats.csv --format csv
```

## Options détaillées

### `perplexity_phrase_sorter.py`
| Option | Description |
|--------|-------------|
| `--input`, `-i` | Fichier texte d'entrée |
| `--text`, `-t` | Texte direct à analyser |
| `--output`, `-o` | Fichier de sortie (optionnel) |
| `--model`, `-m` | Modèle MLX à utiliser (défaut: SmolLM3-3B-4bit) |
| `--verbose`, `-v` | Afficher les détails d'analyse par token |

### `perplexity_batch_processor.py`
| Option | Description |
|--------|-------------|
| `--input`, `-i` | Fichier texte très long à traiter |
| `--output`, `-o` | Fichier de sortie pour les résultats |
| `--database`, `-d` | Fichier base SQLite (défaut: perplexity_cache.db) |
| `--model`, `-m` | Modèle MLX à utiliser |
| `--verbose`, `-v` | Mode verbose avec détails par token |
| `--batch-size`, `-b` | Taille des lots pour sauvegarde (défaut: 100) |
| `--stats-only`, `-s` | Afficher uniquement les statistiques |
| `--results-only`, `-r` | Afficher uniquement les résultats triés |

### `extract_results.py`
| Option | Description |
|--------|-------------|
| `database` | Chemin vers la base SQLite (argument obligatoire) |
| `--output`, `-o` | Fichier de sortie |
| `--format`, `-f` | Format de sortie (standard/csv/json) |
| `--limit`, `-l` | Nombre maximum de phrases à afficher |
| `--top`, `-t` | Top N phrases avec perplexité la plus élevée |
| `--bottom`, `-b` | Top N phrases avec perplexité la plus faible |
| `--complex`, `-c` | **Top N phrases les plus complexes** (perplexité × longueur) |
| `--min-length` | Longueur minimale pour --complex (défaut: 50) |
| `--search`, `-s` | Rechercher des phrases contenant un mot-clé |
| `--min-perplexity` | Filtre: perplexité minimale |
| `--max-perplexity` | Filtre: perplexité maximale |
| `--stats-only` | Afficher uniquement les statistiques |

## Exemples de sortie

### Traitement simple
```
Traitement de 3 phrases...
  Phrase 1/3... ✓
  Phrase 2/3... ✓
  Phrase 3/3... ✓

================================================================================
PHRASES TRIÉES PAR PERPLEXITÉ DÉCROISSANTE
================================================================================
Cette technologie transforme notre société [[892.45]]
L'intelligence artificielle fascine [[456.78]]
Les modèles évoluent rapidement [[234.12]]
```

### Traitement batch avec statistiques
```
==================================================
STATISTIQUES DE LA BASE
==================================================
Total phrases : 45
Phrases traitées : 45
Phrases restantes : 0
Complétude : 100.0%
Perplexité moyenne : 234.56
Perplexité min/max : 12.34 / 892.45
```

### Extraction des phrases les plus complexes
```
================================================================================
TOP 10 - PHRASES LES PLUS COMPLEXES (perplexité × longueur)
================================================================================
  1. L'intelligence artificielle représente l'un des défis technologiques les plus fascinants de notre époque contemporaine
     [Perplexité: 245.67, Complexité: 1245.3, Longueur: 108]

  2. La réglementation peine à suivre le rythme effréné de l'innovation technologique dans ce domaine
     [Perplexité: 189.23, Complexité: 798.4, Longueur: 78]
```

## Interprétation des résultats

### Métriques disponibles

- **Perplexité** : Mesure de "surprise" du modèle face à la phrase
  - *Élevée* : Phrase imprévisible, vocabulaire rare, structure inhabituelle
  - *Faible* : Phrase prévisible, vocabulaire courant, structure simple
  - *∞* : Phrase trop courte ou erreur de calcul

- **Complexité linguistique** : Score combinant perplexité et longueur
  - *Score = Perplexité × log(Longueur)*
  - Privilégie les phrases longues ET surprenantes
  - Évite les phrases courtes avec juste un mot rare

### Différence entre --top et --complex

```bash
# --top : phrases avec perplexité pure la plus élevée
# Peut inclure des phrases courtes avec mots rares
python extract_results.py db.sqlite --top 10

# --complex : phrases linguistiquement complexes
# Combine perplexité élevée + longueur substantielle  
python extract_results.py db.sqlite --complex 10
```

## Structure du projet

```
perplexityphrase/
├── perplexity_phrase_sorter.py     # Traitement simple de petits textes
├── perplexity_batch_processor.py   # Traitement batch avec persistance SQLite
├── extract_results.py              # Extraction et analyse des résultats
├── requirements.txt                # Dépendances Python
├── exemple_texte.txt               # Exemple de texte court
├── long_text_example.txt           # Exemple de texte long pour batch
└── README.md                       # Ce fichier
```

## Fonctionnement technique

Le programme :

1. **Tokenise** chaque phrase avec le tokenizer du modèle
2. **Calcule** la probabilité de chaque token suivant son contexte
3. **Mesure** la log-vraisemblance négative moyenne
4. **Convertit** en perplexité : `exp(-log_likelihood_moyenne)`

En mode verbose, vous pouvez voir le détail token par token :
```
Phrase 1/3:
  Analyse: 'L'intelligence artificielle fascine...'
  Tokens (4): [235, 42156, 98234, 15678]...
    Pos 0: token 42156, prob=0.000234, nll=8.3567
    Pos 1: token 98234, prob=0.123456, nll=2.0923
    Pos 2: token 15678, prob=0.045678, nll=3.0879
  Résultat: NLL moyen=4.5123, Perplexité=91.23
```

## Workflow recommandé

### Pour un petit texte (< 100 phrases)
```bash
python perplexity_phrase_sorter.py --input mon_texte.txt
```

### Pour un long texte (> 100 phrases) 
```bash
# 1. Traitement avec persistance
python perplexity_batch_processor.py --input long_texte.txt

# 2. Analyse des résultats  
python extract_results.py perplexity_cache.db --complex 20
python extract_results.py perplexity_cache.db --search "mot-clé"
```

### Cas d'usage avancés
```bash
# Export pour analyse externe
python extract_results.py perplexity_cache.db --output data.csv --format csv

# Filtrage par plage de perplexité
python extract_results.py perplexity_cache.db --min-perplexity 100 --max-perplexity 500

# Suivi de progression sur un très long traitement
python perplexity_batch_processor.py --database ma_base.db --stats-only
```

## Limitations

- Fonctionne uniquement sur macOS avec Apple Silicon
- Traitement séquentiel (pas de parallélisation car MLX n'est pas thread-safe)
- La vitesse dépend de la longueur des phrases et du nombre de tokens
- Base SQLite : un seul processus à la fois par base

## Licence

MIT License

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir des issues ou proposer des pull requests.