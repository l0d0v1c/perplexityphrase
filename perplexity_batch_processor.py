#!/usr/bin/env python3
"""
Programme pour calculer la perplexité de phrases dans de très longs textes.
Utilise SQLite pour la persistance et permet la reprise après crash.
"""

import re
import math
import sqlite3
import os
import time
import mlx.core as mx
from mlx_lm import load
from typing import List, Tuple, Optional
import argparse
from pathlib import Path


class PerplexityBatchProcessor:
    def __init__(self, model_name: str = "mlx-community/SmolLM3-3B-4bit", db_path: str = "perplexity_cache.db"):
        """Initialise le processeur batch avec base SQLite."""
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Initialiser la base de données
        self.init_database()
        
    def init_database(self):
        """Initialise la base de données SQLite."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT UNIQUE NOT NULL,
                perplexity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index pour les performances
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_text ON sentences (text)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_perplexity ON sentences (perplexity)")
        
        self.conn.commit()
        print(f"Base de données initialisée : {self.db_path}")
    
    def load_model(self):
        """Charge le modèle MLX (fait paresseusement)."""
        if self.model is None:
            print(f"Chargement du modèle {self.model_name}...")
            self.model, self.tokenizer = load(self.model_name)
            print("Modèle chargé avec succès.")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Découpe le texte en phrases."""
        text = text.strip()
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return sentences
    
    def store_sentences(self, sentences: List[str]):
        """Stocke les phrases dans la base."""
        for sentence in sentences:
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO sentences (text, perplexity) VALUES (?, NULL)",
                    (sentence,)
                )
            except sqlite3.Error as e:
                print(f"Erreur SQLite lors du stockage de '{sentence[:50]}...': {e}")
                continue
        
        self.conn.commit()
        print(f"Stocké {len(sentences)} phrases dans la base de données.")
    
    def get_pending_sentences(self) -> List[Tuple[int, str]]:
        """Récupère les phrases pas encore traitées."""
        cursor = self.conn.execute("""
            SELECT id, text 
            FROM sentences 
            WHERE perplexity IS NULL
            ORDER BY id
        """)
        return cursor.fetchall()
    
    def calculate_perplexity(self, sentence: str, verbose: bool = False) -> float:
        """Calcule la perplexité d'une phrase."""
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
            total_nll = 0.0
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
                last_logits = logits[0, -1, :]
                
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
                    total_nll += 20.0
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
            return float('inf')
    
    def store_result(self, sentence_id: int, perplexity: float):
        """Stocke le résultat dans la base."""
        try:
            self.conn.execute(
                "UPDATE sentences SET perplexity = ? WHERE id = ?",
                (perplexity, sentence_id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Erreur lors du stockage du résultat pour sentence_id {sentence_id}: {e}")
    
    def process_batch(self, verbose: bool = False, batch_size: int = 100):
        """Traite un lot de phrases en attente."""
        # Charger le modèle seulement si nécessaire
        self.load_model()
        
        pending = self.get_pending_sentences()
        
        if not pending:
            print("Aucune phrase en attente de traitement.")
            return
        
        total_pending = len(pending)
        print(f"\nTraitement de {total_pending} phrases en attente...")
        
        processed_count = 0
        
        for sentence_id, sentence in pending:
            start_time = time.time()
            
            if verbose:
                print(f"\n--- Phrase {processed_count + 1}/{total_pending} (ID: {sentence_id}) ---")
            else:
                print(f"  Phrase {processed_count + 1}/{total_pending}...", end=" ", flush=True)
            
            perplexity = self.calculate_perplexity(sentence, verbose)
            
            self.store_result(sentence_id, perplexity)
            
            if not verbose:
                print(f"✓ ({perplexity:.2f})")
            
            processed_count += 1
            
            # Commit périodique pour éviter de perdre trop de données
            if processed_count % batch_size == 0:
                print(f"  → Sauvegarde intermédiaire ({processed_count}/{total_pending})")
        
        print(f"\nTraitement terminé : {processed_count} phrases traitées.")
    
    def get_results_sorted(self) -> List[Tuple[str, float]]:
        """Récupère tous les résultats triés par perplexité décroissante."""
        cursor = self.conn.execute("""
            SELECT text, perplexity 
            FROM sentences 
            WHERE perplexity IS NOT NULL
            ORDER BY perplexity DESC
        """)
        return cursor.fetchall()
    
    def print_statistics(self):
        """Affiche les statistiques du traitement."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM sentences")
        total_sentences = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM sentences WHERE perplexity IS NOT NULL")
        processed_sentences = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT AVG(perplexity), MIN(perplexity), MAX(perplexity) FROM sentences WHERE perplexity IS NOT NULL AND perplexity != 'inf'")
        stats = cursor.fetchone()
        
        print(f"\n=== STATISTIQUES ===")
        print(f"Total phrases : {total_sentences}")
        print(f"Phrases traitées : {processed_sentences}")
        print(f"Phrases restantes : {total_sentences - processed_sentences}")
        
        if stats and stats[0] is not None:
            print(f"Perplexité moyenne : {stats[0]:.2f}")
            print(f"Perplexité min/max : {stats[1]:.2f} / {stats[2]:.2f}")
    
    def close(self):
        """Ferme la connexion à la base."""
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Traitement batch de perplexité avec persistance SQLite')
    parser.add_argument('--input', '-i', type=str, required=True, help='Fichier texte très long à traiter')
    parser.add_argument('--output', '-o', type=str, help='Fichier de sortie pour les résultats')
    parser.add_argument('--database', '-d', type=str, default='perplexity_cache.db', help='Fichier base SQLite')
    parser.add_argument('--model', '-m', type=str, default='mlx-community/SmolLM3-3B-4bit', help='Modèle MLX')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbose avec détails par token')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Taille des lots pour sauvegarde')
    parser.add_argument('--stats-only', '-s', action='store_true', help='Afficher uniquement les statistiques')
    parser.add_argument('--results-only', '-r', action='store_true', help='Afficher uniquement les résultats triés')
    
    args = parser.parse_args()
    
    # Initialiser le processeur
    processor = PerplexityBatchProcessor(args.model, args.database)
    
    try:
        if args.stats_only:
            processor.print_statistics()
            return
        
        if args.results_only:
            results = processor.get_results_sorted()
            if not results:
                print("Aucun résultat disponible.")
                return
            
            print(f"\n{'='*80}")
            print("PHRASES TRIÉES PAR PERPLEXITÉ DÉCROISSANTE")
            print(f"{'='*80}")
            
            for sentence, perplexity in results:
                if perplexity == float('inf'):
                    perp_str = "∞"
                else:
                    perp_str = f"{perplexity:.2f}"
                print(f"{sentence} [[{perp_str}]]")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write("PHRASES TRIÉES PAR PERPLEXITÉ DÉCROISSANTE\n")
                    f.write("="*80 + "\n")
                    for sentence, perplexity in results:
                        if perplexity == float('inf'):
                            perp_str = "∞"
                        else:
                            perp_str = f"{perplexity:.2f}"
                        f.write(f"{sentence} [[{perp_str}]]\n")
                print(f"\nRésultats sauvegardés dans {args.output}")
            
            return
        
        # Vérifier si le fichier d'entrée existe
        if not os.path.exists(args.input):
            print(f"Erreur : Le fichier {args.input} n'existe pas.")
            return
        
        # Lire et découper le texte
        print(f"Lecture du fichier : {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Taille du fichier : {len(text)} caractères")
        
        sentences = processor.split_into_sentences(text)
        print(f"Découpage : {len(sentences)} phrases détectées")
        
        # Stocker les phrases
        processor.store_sentences(sentences)
        
        # Afficher les statistiques avant traitement
        processor.print_statistics()
        
        # Traiter les phrases en attente
        processor.process_batch(args.verbose, args.batch_size)
        
        # Afficher les statistiques finales
        processor.print_statistics()
        
        print(f"\nPour voir les résultats triés :")
        print(f"python {__file__} --database {args.database} --results-only")
        
    except KeyboardInterrupt:
        print("\n\nInterruption détectée. Progression sauvegardée dans la base.")
        processor.print_statistics()
    except Exception as e:
        print(f"Erreur : {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.close()


if __name__ == "__main__":
    main()