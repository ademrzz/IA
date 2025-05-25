# genetic_optimizer.py
"""
Algorithme Génétique Principal pour l'optimisation d'emplois du temps
Utilise le réseau de neurones de scoring pour évaluer les solutions
Système IA d'EDT USTHB
"""

import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
import json
from collections import defaultdict

# Imports des modules précédents
from timetable_config import *
from neural_scoring import ScoringNeuralNetwork, FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeneticParameters:
    """Paramètres de l'algorithme génétique"""
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elite_size: int = 5
    tournament_size: int = 3
    stagnation_limit: int = 20
    min_fitness_threshold: float = 0.85
    
    # Paramètres de parallélisation
    use_multiprocessing: bool = True
    n_workers: int = 4

@dataclass
class FitnessMetrics:
    """Métriques détaillées de fitness"""
    score_total: float = 0.0
    contraintes_dures: float = 0.0
    contraintes_souples: float = 0.0
    scoring_neural: float = 0.0
    equilibrage_charge: float = 0.0
    utilisation_ressources: float = 0.0
    satisfaction_pedagogique: float = 0.0
    violations: Dict[str, int] = field(default_factory=dict)

class Individual:
    """Représente un individu (solution) dans la population"""
    
    def __init__(self, data_manager: DataManager, seances: List[Seance]):
        self.data_manager = data_manager
        self.seances = seances
        self.chromosome: Dict[int, Dict] = {}  # seance_id -> {jour, tranche, codloc, id_enseignant}
        self.fitness_metrics = FitnessMetrics()
        self.is_evaluated = False
        self.generation_created = 0
        
        # Structures pour accélération des calculs
        self._occupation_matrix = None
        self._enseignant_schedule = None
        self._groupe_schedule = None
        self._salle_schedule = None
        
        self._initialize_chromosome()
    
    def _initialize_chromosome(self):
        """Initialise le chromosome avec des placements aléatoires valides"""
        for i, seance in enumerate(self.seances):
            placement = self._generate_random_valid_placement(seance)
            if placement:
                self.chromosome[i] = placement
            else:
                # Placement forcé si aucun placement valide trouvé
                self.chromosome[i] = self._generate_forced_placement(seance)
    
    def _generate_random_valid_placement(self, seance: Seance) -> Optional[Dict]:
        """Génère un placement aléatoire valide pour une séance"""
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts:
            jour = random.randint(1, self.data_manager.config.JOURS_SEMAINE)
            tranche = random.randint(1, self.data_manager.config.TRANCHES_JOUR)
            
            # Trouver un local adapté
            locaux_adaptes = self._find_suitable_locals(seance)
            if not locaux_adaptes:
                attempts += 1
                continue
            
            codloc = random.choice(locaux_adaptes)
            
            # Trouver un enseignant compétent
            enseignants_competents = self.data_manager.trouver_enseignant_competent(
                seance.codform, seance.type_seance
            )
            if not enseignants_competents:
                attempts += 1
                continue
            
            id_enseignant = random.choice(enseignants_competents)
            
            # Vérifier les conflits
            placement = {
                'jour': jour,
                'tranche': tranche,
                'codloc': codloc,
                'id_enseignant': id_enseignant
            }
            
            if self._is_placement_valid(seance, placement):
                return placement
            
            attempts += 1
        
        return None
    
    def _generate_forced_placement(self, seance: Seance) -> Dict:
        """Génère un placement forcé (peut violer des contraintes)"""
        locaux = list(self.data_manager.locaux.keys())
        enseignants = list(self.data_manager.enseignants.keys())
        
        return {
            'jour': random.randint(1, self.data_manager.config.JOURS_SEMAINE),
            'tranche': random.randint(1, self.data_manager.config.TRANCHES_JOUR),
            'codloc': random.choice(locaux) if locaux else 'S01',
            'id_enseignant': random.choice(enseignants) if enseignants else 1
        }
    
    def _find_suitable_locals(self, seance: Seance) -> List[str]:
        """Trouve les locaux adaptés pour une séance"""
        effectif = calculer_effectif_groupe(self.data_manager, seance.codgroupe)
        locaux_adaptes = []
        
        for codloc in self.data_manager.locaux:
            if self.data_manager.est_local_adapte(codloc, seance.type_seance, effectif):
                locaux_adaptes.append(codloc)
        
        return locaux_adaptes
    
    def _is_placement_valid(self, seance: Seance, placement: Dict) -> bool:
        """Vérifie si un placement est valide (contraintes dures uniquement)"""
        jour = placement['jour']
        tranche = placement['tranche']
        codloc = placement['codloc']
        id_enseignant = placement['id_enseignant']
        
        # Vérifier les conflits existants dans le chromosome
        for other_placement in self.chromosome.values():
            if (other_placement['jour'] == jour and 
                other_placement['tranche'] == tranche):
                
                # Conflit de salle
                if other_placement['codloc'] == codloc:
                    return False
                
                # Conflit d'enseignant
                if other_placement['id_enseignant'] == id_enseignant:
                    return False
        
        return True
    
    def mutate(self, mutation_rate: float):
        """Applique une mutation à l'individu"""
        if not self.chromosome:
            return
        
        for seance_id in list(self.chromosome.keys()):
            if random.random() < mutation_rate:
                seance = self.seances[seance_id]
                
                # Types de mutation possibles
                mutation_type = random.choice(['local', 'enseignant', 'horaire', 'complete'])
                
                if mutation_type == 'local':
                    self._mutate_local(seance_id, seance)
                elif mutation_type == 'enseignant':
                    self._mutate_enseignant(seance_id, seance)
                elif mutation_type == 'horaire':
                    self._mutate_horaire(seance_id)
                else:
                    self._mutate_complete(seance_id, seance)
        
        self.is_evaluated = False
    
    def _mutate_local(self, seance_id: int, seance: Seance):
        """Mutation du local uniquement"""
        locaux_adaptes = self._find_suitable_locals(seance)
        if locaux_adaptes:
            self.chromosome[seance_id]['codloc'] = random.choice(locaux_adaptes)
    
    def _mutate_enseignant(self, seance_id: int, seance: Seance):
        """Mutation de l'enseignant uniquement"""
        enseignants_competents = self.data_manager.trouver_enseignant_competent(
            seance.codform, seance.type_seance
        )
        if enseignants_competents:
            self.chromosome[seance_id]['id_enseignant'] = random.choice(enseignants_competents)
    
    def _mutate_horaire(self, seance_id: int):
        """Mutation de l'horaire uniquement"""
        self.chromosome[seance_id]['jour'] = random.randint(1, self.data_manager.config.JOURS_SEMAINE)
        self.chromosome[seance_id]['tranche'] = random.randint(1, self.data_manager.config.TRANCHES_JOUR)
    
    def _mutate_complete(self, seance_id: int, seance: Seance):
        """Mutation complète - nouveau placement aléatoire"""
        new_placement = self._generate_random_valid_placement(seance)
        if new_placement:
            self.chromosome[seance_id] = new_placement
    
    def get_fitness(self) -> float:
        """Retourne le score de fitness total"""
        return self.fitness_metrics.score_total
    
    def copy(self) -> 'Individual':
        """Crée une copie profonde de l'individu"""
        new_individual = Individual(self.data_manager, self.seances)
        new_individual.chromosome = copy.deepcopy(self.chromosome)
        new_individual.fitness_metrics = copy.deepcopy(self.fitness_metrics)
        new_individual.is_evaluated = self.is_evaluated
        new_individual.generation_created = self.generation_created
        return new_individual

class ConflictDetector:
    """Détecteur optimisé de conflits"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def detect_all_conflicts(self, individual: Individual) -> Dict[str, List]:
        """Détecte tous les types de conflits dans un individu"""
        conflicts = {
            'salle_conflicts': [],
            'enseignant_conflicts': [],
            'overload_conflicts': [],
            'capacity_conflicts': []
        }
        
        # Grouper les placements par créneau pour détecter les conflits
        creneaux = defaultdict(list)
        for seance_id, placement in individual.chromosome.items():
            key = (placement['jour'], placement['tranche'])
            creneaux[key].append((seance_id, placement))
        
        # Détecter les conflits de ressources
        for creneau, placements in creneaux.items():
            if len(placements) > 1:
                # Conflits de salle
                salles = defaultdict(list)
                enseignants = defaultdict(list)
                
                for seance_id, placement in placements:
                    salles[placement['codloc']].append(seance_id)
                    enseignants[placement['id_enseignant']].append(seance_id)
                
                # Enregistrer les conflits de salle
                for codloc, seances in salles.items():
                    if len(seances) > 1:
                        conflicts['salle_conflicts'].extend(seances)
                
                # Enregistrer les conflits d'enseignant
                for id_ens, seances in enseignants.items():
                    if len(seances) > 1:
                        conflicts['enseignant_conflicts'].extend(seances)
        
        # Détecter les conflits de capacité
        for seance_id, placement in individual.chromosome.items():
            seance = individual.seances[seance_id]
            effectif = calculer_effectif_groupe(self.data_manager, seance.codgroupe)
            local = self.data_manager.locaux.get(placement['codloc'])
            
            if local and local.capacite < effectif:
                conflicts['capacity_conflicts'].append(seance_id)
        
        return conflicts

class FitnessEvaluator:
    """Évaluateur de fitness utilisant le réseau de neurones"""
    
    def __init__(self, data_manager: DataManager, scoring_model: ScoringNeuralNetwork,
                 feature_extractor: FeatureExtractor):
        self.data_manager = data_manager
        self.scoring_model = scoring_model
        self.feature_extractor = feature_extractor
        self.conflict_detector = ConflictDetector(data_manager)
        
        # Poids pour la combinaison des scores
        self.weights = {
            'hard_constraints': 0.4,
            'neural_scoring': 0.3,
            'load_balancing': 0.15,
            'resource_utilization': 0.15
        }
    
    def evaluate(self, individual: Individual) -> FitnessMetrics:
        """Évalue un individu et retourne ses métriques détaillées"""
        if individual.is_evaluated:
            return individual.fitness_metrics
        
        metrics = FitnessMetrics()
        
        # 1. Évaluation des contraintes dures
        metrics.contraintes_dures = self._evaluate_hard_constraints(individual)
        
        # 2. Scoring par réseau de neurones
        metrics.scoring_neural = self._evaluate_neural_scoring(individual)
        
        # 3. Équilibrage de charge
        metrics.equilibrage_charge = self._evaluate_load_balancing(individual)
        
        # 4. Utilisation des ressources
        metrics.utilisation_ressources = self._evaluate_resource_utilization(individual)
        
        # 5. Satisfaction pédagogique
        metrics.satisfaction_pedagogique = self._evaluate_pedagogical_satisfaction(individual)
        
        # Score total pondéré
        metrics.score_total = (
            self.weights['hard_constraints'] * metrics.contraintes_dures +
            self.weights['neural_scoring'] * metrics.scoring_neural +
            self.weights['load_balancing'] * metrics.equilibrage_charge +
            self.weights['resource_utilization'] * metrics.utilisation_ressources
        )
        
        individual.fitness_metrics = metrics
        individual.is_evaluated = True
        
        return metrics
    
    def _evaluate_hard_constraints(self, individual: Individual) -> float:
        """Évalue les contraintes dures (conflits)"""
        conflicts = self.conflict_detector.detect_all_conflicts(individual)
        
        total_violations = (
            len(conflicts['salle_conflicts']) +
            len(conflicts['enseignant_conflicts']) +
            len(conflicts['capacity_conflicts'])
        )
        
        # Pénalisation des violations
        if total_violations == 0:
            return 1.0
        
        penalty = min(total_violations * 0.1, 0.8)
        return max(0.0, 1.0 - penalty)
    
    def _evaluate_neural_scoring(self, individual: Individual) -> float:
        """Évalue en utilisant le réseau de neurones"""
        if not individual.chromosome:
            return 0.0
        
        total_score = 0.0
        valid_placements = 0
        
        # Construire le contexte de l'emploi du temps
        schedule_context = self._build_schedule_context(individual)
        
        for seance_id, placement in individual.chromosome.items():
            try:
                seance = individual.seances[seance_id]
                
                # Extraction des caractéristiques
                features = self.feature_extractor.extract_features(
                    seance,
                    placement['jour'],
                    placement['tranche'],
                    placement['codloc'],
                    placement['id_enseignant'],
                    schedule_context
                )
                
                # Normalisation
                features_normalized = self.feature_extractor.transform_features(features)
                
                # Prédiction du score
                score = self.scoring_model.predict_score(features_normalized)
                total_score += score
                valid_placements += 1
                
            except Exception as e:
                logger.warning(f"Erreur lors du scoring neural pour la séance {seance_id}: {e}")
                continue
        
        return total_score / max(valid_placements, 1)
    
    def _build_schedule_context(self, individual: Individual) -> Dict:
        """Construit le contexte de l'emploi du temps pour le scoring neural"""
        context = {}
        
        for seance_id, placement in individual.chromosome.items():
            seance = individual.seances[seance_id]
            context[f"seance_{seance_id}"] = {
                'codform': seance.codform,
                'jour': placement['jour'],
                'tranche': placement['tranche'],
                'codloc': placement['codloc'],
                'codgroupe': seance.codgroupe,
                'id_enseignant': placement['id_enseignant'],
                'type_seance': seance.type_seance.value
            }
        
        return context
    
    def _evaluate_load_balancing(self, individual: Individual) -> float:
        """Évalue l'équilibrage de charge"""
        # Charge par jour pour les enseignants
        enseignant_charge = defaultdict(lambda: defaultdict(int))
        groupe_charge = defaultdict(lambda: defaultdict(int))
        
        for seance_id, placement in individual.chromosome.items():
            seance = individual.seances[seance_id]
            
            enseignant_charge[placement['id_enseignant']][placement['jour']] += 1
            groupe_charge[seance.codgroupe][placement['jour']] += 1
        
        # Calcul des déséquilibres
        score = 1.0
        
        # Pénalité pour surcharge des enseignants
        for id_ens, charges in enseignant_charge.items():
            for jour, nb_seances in charges.items():
                if nb_seances > self.data_manager.config.MAX_HEURES_JOUR_ENSEIGNANT:
                    score -= 0.1
        
        # Pénalité pour surcharge des groupes
        for codgroupe, charges in groupe_charge.items():
            for jour, nb_seances in charges.items():
                if nb_seances > self.data_manager.config.MAX_HEURES_JOUR_ETUDIANT:
                    score -= 0.1
        
        return max(0.0, score)
    
    def _evaluate_resource_utilization(self, individual: Individual) -> float:
        """Évalue l'utilisation des ressources"""
        # Utilisation des salles
        salle_utilisation = defaultdict(int)
        
        for placement in individual.chromosome.values():
            salle_utilisation[placement['codloc']] += 1
        
        # Score basé sur la répartition équitable
        nb_salles_utilisees = len(salle_utilisation)
        nb_salles_total = len(self.data_manager.locaux)
        
        utilisation_ratio = nb_salles_utilisees / max(nb_salles_total, 1)
        return min(utilisation_ratio * 1.2, 1.0)
    
    def _evaluate_pedagogical_satisfaction(self, individual: Individual) -> float:
        """Évalue la satisfaction pédagogique"""
        # Cette fonction peut intégrer des règles pédagogiques spécifiques
        # Pour l'instant, score neutre
        return 0.7

class GeneticOperators:
    """Opérateurs génétiques pour la reproduction et mutation"""
    
    @staticmethod
    def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
        """Sélection par tournoi"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.get_fitness())
    
    @staticmethod
    def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Croisement uniforme entre deux parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Croisement uniforme
        for seance_id in parent1.chromosome:
            if seance_id in parent2.chromosome and random.random() < 0.5:
                child1.chromosome[seance_id] = copy.deepcopy(parent2.chromosome[seance_id])
                child2.chromosome[seance_id] = copy.deepcopy(parent1.chromosome[seance_id])
        
        # Marquer comme non évalués
        child1.is_evaluated = False
        child2.is_evaluated = False
        
        return child1, child2
    
    @staticmethod
    def elitism_selection(population: List[Individual], elite_size: int) -> List[Individual]:
        """Sélection élitiste"""
        population.sort(key=lambda ind: ind.get_fitness(), reverse=True)
        return population[:elite_size]

class TimetableGeneticOptimizer:
    """Optimiseur génétique principal pour l'emploi du temps"""
    
    def __init__(self, data_manager: DataManager, scoring_model: ScoringNeuralNetwork,
                 feature_extractor: FeatureExtractor, params: GeneticParameters = None):
        self.data_manager = data_manager
        self.scoring_model = scoring_model
        self.feature_extractor = feature_extractor
        self.params = params or GeneticParameters()
        
        self.evaluator = FitnessEvaluator(data_manager, scoring_model, feature_extractor)
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history = []
        self.stagnation_counter = 0
        
        # Statistiques
        self.stats = {
            'start_time': None,
            'evaluations_count': 0,
            'best_fitness_per_generation': [],
            'average_fitness_per_generation': []
        }
    
    def optimize(self, seances: List[Seance], max_time: Optional[int] = None) -> Individual:
        """
        Lance l'optimisation génétique
        
        Args:
            seances: Liste des séances à placer
            max_time: Temps maximum en secondes (optionnel)
        
        Returns:
            Meilleur individu trouvé
        """
        logger.info("=== Début de l'optimisation génétique ===")
        self.stats['start_time'] = time.time()
        
        # Initialisation de la population
        self._initialize_population(seances)
        
        # Évaluation initiale
        self._evaluate_population()
        self._update_statistics()
        
        logger.info(f"Population initiale - Meilleur fitness: {self.best_individual.get_fitness():.3f}")
        
        # Boucle évolutionnaire
        while not self._should_terminate(max_time):
            self.generation += 1
            
            # Nouvelle génération
            new_population = self._create_new_generation()
            
            # Évaluation
            self.population = new_population
            self._evaluate_population()
            self._update_statistics()
            
            # Logging
            if self.generation % 10 == 0:
                self._log_progress()
        
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"=== Optimisation terminée en {elapsed_time:.1f}s ===")
        logger.info(f"Générations: {self.generation}, Évaluations: {self.stats['evaluations_count']}")
        logger.info(f"Meilleur fitness: {self.best_individual.get_fitness():.3f}")
        
        return self.best_individual
    
    def _initialize_population(self, seances: List[Seance]):
        """Initialise la population"""
        logger.info(f"Initialisation de la population ({self.params.population_size} individus)")
        
        self.population = []
        for i in range(self.params.population_size):
            individual = Individual(self.data_manager, seances)
            individual.generation_created = 0
            self.population.append(individual)
        
        logger.info(f"Population initialisée avec {len(seances)} séances par individu")
    
    def _evaluate_population(self):
        """Évalue toute la population"""
        if self.params.use_multiprocessing and self.params.n_workers > 1:
            self._evaluate_population_parallel()
        else:
            self._evaluate_population_sequential()
        
        # Mise à jour du meilleur individu
        current_best = max(self.population, key=lambda ind: ind.get_fitness())
        if self.best_individual is None or current_best.get_fitness() > self.best_individual.get_fitness():
            self.best_individual = current_best.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
    
    def _evaluate_population_sequential(self):
        """Évaluation séquentielle de la population"""
        for individual in self.population:
            if not individual.is_evaluated:
                self.evaluator.evaluate(individual)
                self.stats['evaluations_count'] += 1
    
    def _evaluate_population_parallel(self):
        """Évaluation parallèle de la population"""
        non_evaluated = [ind for ind in self.population if not ind.is_evaluated]
        
        if not non_evaluated:
            return
        
        with ThreadPoolExecutor(max_workers=self.params.n_workers) as executor:
            futures = [executor.submit(self.evaluator.evaluate, ind) for ind in non_evaluated]
            for future in futures:
                future.result()
                self.stats['evaluations_count'] += 1
    
    def _create_new_generation(self) -> List[Individual]:
        """Crée une nouvelle génération"""
        new_population = []
        
        # Élitisme
        elite = GeneticOperators.elitism_selection(self.population, self.params.elite_size)
        new_population.extend([ind.copy() for ind in elite])
        
        # Reproduction
        while len(new_population) < self.params.population_size:
            # Sélection des parents
            parent1 = GeneticOperators.tournament_selection(self.population, self.params.tournament_size)
            parent2 = GeneticOperators.tournament_selection(self.population, self.params.tournament_size)
            
            # Croisement
            if random.random() < self.params.crossover_rate:
                child1, child2 = GeneticOperators.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1.mutate(self.params.mutation_rate)
            child2.mutate(self.params.mutation_rate)
            
            # Ajout à la nouvelle population
            child1.generation_created = self.generation
            child2.generation_created = self.generation
            
            new_population.extend([child1, child2])
        
        # Truncature si nécessaire
        return new_population[:self.params.population_size]
    
    def _should_terminate(self, max_time: Optional[int] = None) -> bool:
        """Vérifie les conditions d'arrêt"""
        # Limite de générations
        if self.generation >= self.params.max_generations:
            logger.info("Arrêt: limite de générations atteinte")
            return True
        
        # Limite de temps
        if max_time and (time.time() - self.stats['start_time']) > max_time:
            logger.info("Arrêt: limite de temps atteinte")
            return True
        
        # Fitness satisfaisant
        if (self.best_individual and 
            self.best_individual.get_fitness() >= self.params.min_fitness_threshold):
            logger.info("Arrêt: fitness satisfaisant atteint")
            return True
        
        # Stagnation
        if self.stagnation_counter >= self.params.stagnation_limit:
            logger.info("Arrêt: stagnation détectée")
            return True
        
        return False
    
    def _update_statistics(self):
        """Met à jour les statistiques"""
        fitnesses = [ind.get_fitness() for ind in self.population]
        
        self.stats['best_fitness_per_generation'].append(max(fitnesses))
        self.stats['average_fitness_per_generation'].append(np.mean(fitnesses))
        
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses)
        })
    
    def _log_progress(self):
        """Log du progrès"""
        current_best = max(self.population, key=lambda ind: ind.get_fitness())
        avg_fitness = np.mean([ind.get_fitness() for ind in self.population])
        
        logger.info(f"Génération {self.generation}: "
                   f"Meilleur={current_best.get_fitness():.3f}, "
                   f"Moyenne={avg_fitness:.3f}, "
                   f"Stagnation={self.stagnation_counter}")
    
    def get_solution_summary(self) -> Dict:
        """Retourne un résumé de la meilleure solution"""
        if not self.best_individual:
            return {}
        
        conflicts = self.evaluator.conflict_detector.detect_all_conflicts(self.best_individual)
        
        return {
            'fitness': self.best_individual.get_fitness(),
            'metrics': self.best_individual.fitness_metrics.__dict__,
            'conflicts': {k: len(v) for k, v in conflicts.items()},
            'total_seances': len(self.best_individual.chromosome),
            'generation_found': self.best_individual.generation_created,
            'optimization_stats': self.stats
        }
    
    def export_solution(self, filepath: str):
        """Exporte la meilleure solution"""
        if not self.best_individual:
            logger.warning("Aucune solution à exporter")
            return
        
        solution_data = {
            'solution_summary': self.get_solution_summary(),
            'chromosome': self.best_individual.chromosome,
            'fitness_