#!/usr/bin/env python3
"""
Script pour extraire les résultats d'une base SQLite de perplexité
et afficher les phrases triées par perplexité décroissante.
"""

import sqlite3
import argparse
import os
from typing import List, Tuple, Optional


class PerplexityExtractor:
    def __init__(self, db_path: str):
        """Initialise l'extracteur avec le chemin de la base."""
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"La base de données {db_path} n'existe pas.")
        
        self.conn = sqlite3.connect(db_path)
    
    def get_statistics(self) -> dict:
        """Récupère les statistiques de la base."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM sentences")
        total_sentences = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM sentences WHERE perplexity IS NOT NULL")
        processed_sentences = cursor.fetchone()[0]
        
        cursor = self.conn.execute("""
            SELECT AVG(perplexity), MIN(perplexity), MAX(perplexity) 
            FROM sentences 
            WHERE perplexity IS NOT NULL AND perplexity != 'inf'
        """)
        stats = cursor.fetchone()
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM sentences WHERE perplexity = 'inf'")
        infinite_count = cursor.fetchone()[0]
        
        return {
            'total': total_sentences,
            'processed': processed_sentences,
            'remaining': total_sentences - processed_sentences,
            'avg_perplexity': stats[0] if stats[0] is not None else None,
            'min_perplexity': stats[1] if stats[1] is not None else None,
            'max_perplexity': stats[2] if stats[2] is not None else None,
            'infinite_count': infinite_count
        }
    
    def get_sentences_by_perplexity(self, limit: Optional[int] = None, 
                                   min_perplexity: Optional[float] = None,
                                   max_perplexity: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Récupère les phrases triées par perplexité décroissante.
        
        Args:
            limit: Nombre maximum de phrases à récupérer
            min_perplexity: Perplexité minimale (filtre)
            max_perplexity: Perplexité maximale (filtre)
        """
        query = "SELECT text, perplexity FROM sentences WHERE perplexity IS NOT NULL"
        params = []
        
        # Filtres
        if min_perplexity is not None:
            query += " AND perplexity >= ?"
            params.append(min_perplexity)
        
        if max_perplexity is not None:
            query += " AND perplexity <= ?"
            params.append(max_perplexity)
        
        # Tri par perplexité décroissante
        query += " ORDER BY perplexity DESC"
        
        # Limite
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.conn.execute(query, params)
        return cursor.fetchall()
    
    def get_top_perplexity_sentences(self, n: int = 10) -> List[Tuple[str, float]]:
        """Récupère les N phrases avec la plus haute perplexité."""
        return self.get_sentences_by_perplexity(limit=n)
    
    def get_most_complex_sentences(self, n: int = 10, min_length: int = 50) -> List[Tuple[str, float, float]]:
        """
        Récupère les N phrases les plus complexes (perplexité élevée + longueur substantielle).
        
        Args:
            n: Nombre de phrases à retourner
            min_length: Longueur minimale en caractères
            
        Returns:
            Liste de tuples (text, perplexity, complexity_score)
        """
        cursor = self.conn.execute("""
            SELECT text, perplexity, LENGTH(text) as length,
                   (perplexity * LOG(LENGTH(text))) as complexity_score
            FROM sentences 
            WHERE perplexity IS NOT NULL 
              AND perplexity != 'inf' 
              AND LENGTH(text) >= ?
            ORDER BY complexity_score DESC 
            LIMIT ?
        """, (min_length, n))
        
        results = cursor.fetchall()
        return [(text, perplexity, complexity_score) for text, perplexity, length, complexity_score in results]
    
    def get_sentences_by_complexity(self, limit: Optional[int] = None, 
                                   min_length: int = 30,
                                   complexity_weight: float = 1.0) -> List[Tuple[str, float, float]]:
        """
        Récupère les phrases triées par score de complexité.
        
        Args:
            limit: Nombre maximum de phrases
            min_length: Longueur minimale en caractères
            complexity_weight: Poids pour ajuster l'importance de la longueur
            
        Returns:
            Liste de tuples (text, perplexity, complexity_score)
        """
        query = """
            SELECT text, perplexity, LENGTH(text) as length,
                   (perplexity * POWER(LENGTH(text), ?)) as complexity_score
            FROM sentences 
            WHERE perplexity IS NOT NULL 
              AND perplexity != 'inf' 
              AND LENGTH(text) >= ?
            ORDER BY complexity_score DESC
        """
        params = [complexity_weight, min_length]
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.conn.execute(query, params)
        results = cursor.fetchall()
        return [(text, perplexity, complexity_score) for text, perplexity, length, complexity_score in results]
    
    def get_bottom_perplexity_sentences(self, n: int = 10) -> List[Tuple[str, float]]:
        """Récupère les N phrases avec la plus faible perplexité."""
        cursor = self.conn.execute("""
            SELECT text, perplexity 
            FROM sentences 
            WHERE perplexity IS NOT NULL AND perplexity != 'inf'
            ORDER BY perplexity ASC 
            LIMIT ?
        """, (n,))
        return cursor.fetchall()
    
    def search_sentences(self, keyword: str, case_sensitive: bool = False) -> List[Tuple[str, float]]:
        """Recherche des phrases contenant un mot-clé."""
        if case_sensitive:
            query = """
                SELECT text, perplexity 
                FROM sentences 
                WHERE text LIKE ? AND perplexity IS NOT NULL
                ORDER BY perplexity DESC
            """
            pattern = f"%{keyword}%"
        else:
            query = """
                SELECT text, perplexity 
                FROM sentences 
                WHERE LOWER(text) LIKE LOWER(?) AND perplexity IS NOT NULL
                ORDER BY perplexity DESC
            """
            pattern = f"%{keyword}%"
        
        cursor = self.conn.execute(query, (pattern,))
        return cursor.fetchall()
    
    def export_to_text(self, output_path: str, format_type: str = "standard"):
        """Exporte les résultats vers un fichier texte."""
        sentences = self.get_sentences_by_perplexity()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if format_type == "standard":
                f.write("PHRASES TRIÉES PAR PERPLEXITÉ DÉCROISSANTE\n")
                f.write("=" * 80 + "\n\n")
                
                for sentence, perplexity in sentences:
                    if perplexity == float('inf'):
                        perp_str = "∞"
                    else:
                        perp_str = f"{perplexity:.2f}"
                    f.write(f"{sentence} [[{perp_str}]]\n")
            
            elif format_type == "csv":
                f.write("sentence,perplexity\n")
                for sentence, perplexity in sentences:
                    # Échapper les guillemets dans le texte
                    escaped_sentence = sentence.replace('"', '""')
                    f.write(f'"{escaped_sentence}",{perplexity}\n')
            
            elif format_type == "json":
                import json
                data = [{"sentence": sentence, "perplexity": perplexity} 
                       for sentence, perplexity in sentences]
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def close(self):
        """Ferme la connexion à la base."""
        if hasattr(self, 'conn'):
            self.conn.close()


def print_statistics(stats: dict):
    """Affiche les statistiques formatées."""
    print(f"\n{'='*50}")
    print("STATISTIQUES DE LA BASE")
    print(f"{'='*50}")
    print(f"Total phrases : {stats['total']}")
    print(f"Phrases traitées : {stats['processed']}")
    print(f"Phrases restantes : {stats['remaining']}")
    
    if stats['processed'] > 0:
        completion = (stats['processed'] / stats['total']) * 100
        print(f"Complétude : {completion:.1f}%")
    
    if stats['avg_perplexity'] is not None:
        print(f"Perplexité moyenne : {stats['avg_perplexity']:.2f}")
        print(f"Perplexité min/max : {stats['min_perplexity']:.2f} / {stats['max_perplexity']:.2f}")
    
    if stats['infinite_count'] > 0:
        print(f"Phrases avec perplexité infinie : {stats['infinite_count']}")


def print_sentences(sentences, title: str = "RÉSULTATS", show_complexity: bool = False):
    """Affiche les phrases formatées."""
    if not sentences:
        print("Aucune phrase trouvée.")
        return
    
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    
    for i, item in enumerate(sentences, 1):
        if show_complexity and len(item) == 3:
            # Format avec score de complexité
            sentence, perplexity, complexity_score = item
            if perplexity == float('inf'):
                perp_str = "∞"
            else:
                perp_str = f"{perplexity:.2f}"
            print(f"{i:3d}. {sentence}")
            print(f"     [Perplexité: {perp_str}, Complexité: {complexity_score:.1f}, Longueur: {len(sentence)}]")
        else:
            # Format standard
            sentence, perplexity = item[:2]
            if perplexity == float('inf'):
                perp_str = "∞"
            else:
                perp_str = f"{perplexity:.2f}"
            print(f"{i:3d}. {sentence} [[{perp_str}]]")


def main():
    parser = argparse.ArgumentParser(description='Extrait les résultats de perplexité depuis une base SQLite')
    parser.add_argument('database', help='Chemin vers la base SQLite')
    parser.add_argument('--output', '-o', help='Fichier de sortie')
    parser.add_argument('--format', '-f', choices=['standard', 'csv', 'json'], default='standard',
                       help='Format de sortie (défaut: standard)')
    parser.add_argument('--limit', '-l', type=int, help='Nombre maximum de phrases à afficher')
    parser.add_argument('--top', '-t', type=int, help='Afficher les N phrases avec la plus haute perplexité')
    parser.add_argument('--bottom', '-b', type=int, help='Afficher les N phrases avec la plus faible perplexité')
    parser.add_argument('--complex', '-c', type=int, help='Afficher les N phrases les plus complexes (perplexité + longueur)')
    parser.add_argument('--min-length', type=int, default=50, help='Longueur minimale pour --complex (défaut: 50)')
    parser.add_argument('--min-perplexity', type=float, help='Perplexité minimale (filtre)')
    parser.add_argument('--max-perplexity', type=float, help='Perplexité maximale (filtre)')
    parser.add_argument('--search', '-s', help='Rechercher des phrases contenant ce mot-clé')
    parser.add_argument('--case-sensitive', action='store_true', help='Recherche sensible à la casse')
    parser.add_argument('--stats-only', action='store_true', help='Afficher uniquement les statistiques')
    
    args = parser.parse_args()
    
    try:
        # Initialiser l'extracteur
        extractor = PerplexityExtractor(args.database)
        
        # Afficher les statistiques
        stats = extractor.get_statistics()
        print_statistics(stats)
        
        if args.stats_only:
            return
        
        # Récupérer les phrases selon les critères
        sentences = []
        title = "PHRASES TRIÉES PAR PERPLEXITÉ DÉCROISSANTE"
        show_complexity = False
        
        if args.top:
            sentences = extractor.get_top_perplexity_sentences(args.top)
            title = f"TOP {args.top} - PERPLEXITÉ LA PLUS ÉLEVÉE"
        
        elif args.bottom:
            sentences = extractor.get_bottom_perplexity_sentences(args.bottom)
            title = f"TOP {args.bottom} - PERPLEXITÉ LA PLUS FAIBLE"
        
        elif args.complex:
            sentences = extractor.get_most_complex_sentences(args.complex, args.min_length)
            title = f"TOP {args.complex} - PHRASES LES PLUS COMPLEXES (perplexité × longueur)"
            show_complexity = True
        
        elif args.search:
            sentences = extractor.search_sentences(args.search, args.case_sensitive)
            title = f"RECHERCHE: '{args.search}' - TRIÉE PAR PERPLEXITÉ"
        
        else:
            sentences = extractor.get_sentences_by_perplexity(
                limit=args.limit,
                min_perplexity=args.min_perplexity,
                max_perplexity=args.max_perplexity
            )
            if args.limit:
                title += f" (LIMITE: {args.limit})"
        
        # Afficher ou sauvegarder les résultats
        if args.output:
            extractor.export_to_text(args.output, args.format)
            print(f"\nRésultats sauvegardés dans {args.output} (format: {args.format})")
        else:
            print_sentences(sentences, title, show_complexity)
        
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
    except Exception as e:
        print(f"Erreur : {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'extractor' in locals():
            extractor.close()


if __name__ == "__main__":
    main()