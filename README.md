# PerplexityPhrase

Programme Python qui analyse la perplexité des phrases d'un texte et les trie par ordre décroissant de complexité linguistique.

## Description

Ce programme utilise le framework MLX d'Apple avec le modèle de langage SmolLM3-3B-4bit pour :

1. **Découper** un texte en phrases
2. **Calculer** la perplexité de chaque phrase (mesure de surprise/complexité pour le modèle)
3. **Trier** les phrases par perplexité décroissante
4. **Afficher** les résultats au format `phrase [[perplexité]]`

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

## Utilisation

### Analyse d'un fichier texte
```bash
python perplexity_phrase_sorter.py --input exemple_texte.txt
```

### Analyse d'un texte direct
```bash
python perplexity_phrase_sorter.py --text "L'intelligence artificielle fascine. Les modèles évoluent rapidement. Cette technologie transforme notre société."
```

### Mode verbose (avec détails par token)
```bash
python perplexity_phrase_sorter.py --input exemple_texte.txt --verbose
```

### Sauvegarder le résultat
```bash
python perplexity_phrase_sorter.py --input exemple_texte.txt --output resultat.txt
```

### Utiliser un autre modèle MLX
```bash
python perplexity_phrase_sorter.py --input exemple_texte.txt --model mlx-community/autre-modele
```

## Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Fichier texte d'entrée |
| `--text`, `-t` | Texte direct à analyser |
| `--output`, `-o` | Fichier de sortie (optionnel) |
| `--model`, `-m` | Modèle MLX à utiliser (défaut: SmolLM3-3B-4bit) |
| `--verbose`, `-v` | Afficher les détails d'analyse par token |
| `--help`, `-h` | Afficher l'aide |

## Exemple de sortie

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

## Interprétation des résultats

- **Perplexité élevée** : Phrase surprenante, complexe ou inhabituelle pour le modèle
- **Perplexité faible** : Phrase prévisible, simple ou courante
- **∞** : Phrase trop courte ou erreur de calcul

## Structure du projet

```
perplexityphrase/
├── perplexity_phrase_sorter.py  # Programme principal
├── requirements.txt             # Dépendances Python
├── exemple_texte.txt           # Exemple de texte à analyser
└── README.md                   # Ce fichier
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

## Limitations

- Fonctionne uniquement sur macOS avec Apple Silicon
- Traitement séquentiel (pas de parallélisation car MLX n'est pas thread-safe)
- La vitesse dépend de la longueur des phrases et du nombre de tokens

## Licence

MIT License

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir des issues ou proposer des pull requests.