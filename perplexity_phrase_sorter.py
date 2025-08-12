#!/usr/bin/env python3
"""
Programme pour calculer la perplexité de phrases et les trier par ordre décroissant.
Utilise MLX et SmolLM3-3B-4bit avec support de parallélisation.
"""

import re
import math
import mlx.core as mx
from mlx_lm import load
from typing import List, Tuple
import argparse


class PerplexityCalculator:
    def __init__(self, model_name: str = "mlx-community/SmolLM3-3B-4bit"):
        """Initialise le calculateur de perplexité avec le modèle MLX."""
        print(f"Chargement du modèle {model_name}...")
        self.model, self.tokenizer = load(model_name)
        print("Modèle chargé avec succès.")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Découpe le texte en phrases."""
        text = text.strip()
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def calculate_perplexity_simple(self, sentence: str, verbose: bool = False) -> float:
        """Calcule la perplexité d'une phrase avec une approche simplifiée."""
        try:
            if verbose:
                print(f"  Analyse: '{sentence[:50]}...'")
            
            # Tokenisation
            tokens = self.tokenizer.encode(sentence)
            if verbose:
                print(f"  Tokens ({len(tokens)}): {tokens[:10]}...")
            
            if len(tokens) < 2:
                if verbose:
                    print("  Phrase trop courte")
                return float('inf')
            
            # Calcul de la log-vraisemblance totale
            total_nll = 0.0  # negative log likelihood
            num_predictions = 0
            
            # Pour chaque position, on prédit le token suivant
            for i in range(len(tokens) - 1):
                context_tokens = tokens[:i+1]
                target_token = tokens[i+1]
                
                # Conversion en tenseurs MLX
                context_ids = mx.array([context_tokens])
                
                # Forward pass
                logits = self.model(context_ids)
                
                # Logits pour le dernier token du contexte
                last_logits = logits[0, -1, :]  # [vocab_size]
                
                # Calcul des probabilités
                probs = mx.softmax(last_logits, axis=0)
                
                # Probabilité du token cible
                target_prob = float(probs[target_token])
                
                if target_prob > 0 and not math.isinf(target_prob) and not math.isnan(target_prob):
                    nll = -math.log(target_prob)
                    total_nll += nll
                    num_predictions += 1
                    if verbose:
                        print(f"    Pos {i}: token {target_token}, prob={target_prob:.6f}, nll={nll:.4f}")
                else:
                    if verbose:
                        print(f"    Pos {i}: token {target_token}, prob invalide: {target_prob}")
                    total_nll += 20.0  # Valeur élevée pour prob très faible
                    num_predictions += 1
            
            if num_predictions == 0:
                if verbose:
                    print("  Aucune prédiction valide")
                return float('inf')
            
            # Perplexité moyenne
            avg_nll = total_nll / num_predictions
            perplexity = math.exp(avg_nll)
            
            if verbose:
                print(f"  Résultat: NLL moyen={avg_nll:.4f}, Perplexité={perplexity:.2f}")
            
            return perplexity
            
        except Exception as e:
            if verbose:
                print(f"  ERREUR: {e}")
                import traceback
                traceback.print_exc()
            return float('inf')
    
    def process_text(self, text: str, verbose: bool = False) -> List[Tuple[str, float]]:
        """Traite un texte complet et retourne les phrases avec leur perplexité."""
        sentences = self.split_into_sentences(text)
        results = []
        
        print(f"\nTraitement de {len(sentences)} phrases...")
        if verbose:
            print("="*60)
        
        # Traitement séquentiel (MLX n'est pas thread-safe)
        for i, sentence in enumerate(sentences, 1):
            if verbose:
                print(f"\nPhrase {i}/{len(sentences)}:")
            else:
                print(f"  Phrase {i}/{len(sentences)}...", end=" ", flush=True)
            
            perplexity = self.calculate_perplexity_simple(sentence, verbose)
            results.append((sentence, perplexity))
            
            if not verbose:
                print("✓")
        
        return results
    
    def sort_by_perplexity(self, sentence_perplexities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Trie les phrases par perplexité décroissante."""
        return sorted(sentence_perplexities, key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description='Calculer la perplexité des phrases et les trier')
    parser.add_argument('--input', '-i', type=str, help='Fichier texte d\'entrée')
    parser.add_argument('--text', '-t', type=str, help='Texte direct à analyser')
    parser.add_argument('--output', '-o', type=str, help='Fichier de sortie (optionnel)')
    parser.add_argument('--model', '-m', type=str, default='mlx-community/SmolLM3-3B-4bit',
                        help='Nom du modèle MLX à utiliser')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Afficher les détails d\'analyse par token')
    
    args = parser.parse_args()
    
    # Récupération du texte d'entrée
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    elif args.text:
        input_text = args.text
    else:
        print("Veuillez fournir un texte via --input ou --text")
        return
    
    # Initialisation du calculateur
    calculator = PerplexityCalculator(args.model)
    
    # Traitement du texte
    sentence_perplexities = calculator.process_text(input_text, args.verbose)
    
    # Tri par perplexité décroissante
    sorted_sentences = calculator.sort_by_perplexity(sentence_perplexities)
    
    # Génération du résultat
    result_lines = []
    result_lines.append("\n" + "="*80)
    result_lines.append("PHRASES TRIÉES PAR PERPLEXITÉ DÉCROISSANTE")
    result_lines.append("="*80)
    
    for sentence, perplexity in sorted_sentences:
        if perplexity == float('inf'):
            perp_str = "∞"
        else:
            perp_str = f"{perplexity:.2f}"
        result_lines.append(f"{sentence} [[{perp_str}]]")
    
    result_text = '\n'.join(result_lines)
    
    # Sortie
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"\nRésultat sauvegardé dans {args.output}")
    else:
        print(result_text)


if __name__ == "__main__":
    main()