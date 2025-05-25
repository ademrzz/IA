# neural_scoring.py
"""
Réseau de Neurones pour le Scoring - Système IA d'EDT USTHB
Évalue la qualité d'un placement de séance dans un emploi du temps
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Import des configurations de base
from timetable_config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScoringFeatures:
    """Structure pour les caractéristiques d'entrée du réseau de scoring"""
    # Caractéristiques de la séance
    type_cours: int  # 0=Cours, 1=TD, 2=TP
    duree_seance: float  # En heures
    effectif_groupe: int
    
    # Caractéristiques du local
    capacite_local: int
    type_local: int  # 0=Amphi, 1=Salle, 2=TD, 3=TP
    ratio_capacite_effectif: float
    
    # Contexte temporel
    jour_semaine: int  # 1-6 (Samedi à Jeudi)
    tranche_horaire: int  # 1-6
    position_jour: float  # 0-1 (début à fin de journée)
    
    # Contraintes d'adjacence
    seances_adjacentes: int  # Nombre de séances avant/après
    meme_module_jour: int  # Autres séances du même module le même jour
    charge_jour_groupe: int  # Nombre total de séances du groupe ce jour
    
    # Profil enseignant
    preference_matin: float  # 0-1
    preference_aprem: float  # 0-1
    charge_jour_enseignant: int
    charge_semaine_enseignant: int
    
    # Utilisation des ressources
    utilisation_salle_jour: float  # Taux d'occupation de la salle ce jour
    dispersion_geographique: float  # Distance moyenne avec autres cours du groupe
    
    # Cohérence pédagogique
    sequence_pedagogique: float  # Respect de l'ordre Cours->TD->TP
    espacement_seances_module: int  # Jours entre séances du même module

class FeatureExtractor:
    """Extracteur de caractéristiques pour le réseau de neurones"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, seance: Seance, jour: int, tranche: int, 
                        codloc: str, id_enseignant: int, 
                        current_schedule: Dict) -> np.ndarray:
        """
        Extrait les caractéristiques d'une séance placée à un créneau donné
        """
        features = []
        
        # 1. Caractéristiques de la séance
        type_mapping = {TypeSeance.COURS: 0, TypeSeance.TD: 1, TypeSeance.TP: 2}
        features.append(type_mapping[seance.type_seance])
        features.append(seance.duree)
        
        effectif = calculer_effectif_groupe(self.data_manager, seance.codgroupe)
        features.append(effectif)
        
        # 2. Caractéristiques du local
        local = self.data_manager.locaux[codloc]
        features.append(local.capacite)
        
        type_local_mapping = {TypeLocal.AMPHI: 0, TypeLocal.SALLE: 1, 
                             TypeLocal.TD: 2, TypeLocal.TP: 3}
        features.append(type_local_mapping[local.type])
        
        # Ratio capacité/effectif
        ratio_capacite = local.capacite / max(effectif, 1)
        features.append(min(ratio_capacite, 5.0))  # Cap à 5 pour éviter les valeurs extrêmes
        
        # 3. Contexte temporel
        features.append(jour)
        features.append(tranche)
        features.append(tranche / 6.0)  # Position normalisée dans la journée
        
        # 4. Contraintes d'adjacence
        adjacents = self._compter_seances_adjacentes(jour, tranche, current_schedule)
        features.append(adjacents)
        
        meme_module = self._compter_meme_module_jour(seance.codform, jour, current_schedule)
        features.append(meme_module)
        
        charge_jour = self._calculer_charge_jour_groupe(seance.codgroupe, jour, current_schedule)
        features.append(charge_jour)
        
        # 5. Profil enseignant
        if id_enseignant in self.data_manager.enseignants:
            enseignant = self.data_manager.enseignants[id_enseignant]
            features.append(enseignant.preference_matin)
            features.append(enseignant.preference_aprem)
        else:
            features.extend([0.5, 0.5])  # Valeurs neutres par défaut
        
        charge_jour_ens = self._calculer_charge_jour_enseignant(id_enseignant, jour, current_schedule)
        features.append(charge_jour_ens)
        
        charge_semaine_ens = self._calculer_charge_semaine_enseignant(id_enseignant, current_schedule)
        features.append(charge_semaine_ens)
        
        # 6. Utilisation des ressources
        utilisation_salle = self._calculer_utilisation_salle_jour(codloc, jour, current_schedule)
        features.append(utilisation_salle)
        
        dispersion = self._calculer_dispersion_geographique(seance.codgroupe, codloc, jour, current_schedule)
        features.append(dispersion)
        
        # 7. Cohérence pédagogique
        sequence_score = self._evaluer_sequence_pedagogique(seance, jour, current_schedule)
        features.append(sequence_score)
        
        espacement = self._calculer_espacement_seances_module(seance.codform, jour, current_schedule)
        features.append(espacement)
        
        return np.array(features, dtype=np.float32)
    
    def _compter_seances_adjacentes(self, jour: int, tranche: int, schedule: Dict) -> int:
        """Compte les séances adjacentes (tranche précédente et suivante)"""
        count = 0
        
        # Vérifier tranche précédente
        if tranche > 1:
            for seance in schedule.values():
                if seance.get('jour') == jour and seance.get('tranche') == tranche - 1:
                    count += 1
        
        # Vérifier tranche suivante
        if tranche < 6:
            for seance in schedule.values():
                if seance.get('jour') == jour and seance.get('tranche') == tranche + 1:
                    count += 1
        
        return count
    
    def _compter_meme_module_jour(self, codform: str, jour: int, schedule: Dict) -> int:
        """Compte les autres séances du même module le même jour"""
        count = 0
        for seance in schedule.values():
            if (seance.get('codform') == codform and 
                seance.get('jour') == jour):
                count += 1
        return count
    
    def _calculer_charge_jour_groupe(self, codgroupe: int, jour: int, schedule: Dict) -> int:
        """Calcule la charge totale du groupe pour un jour donné"""
        count = 0
        for seance in schedule.values():
            if (seance.get('codgroupe') == codgroupe and 
                seance.get('jour') == jour):
                count += 1
        return count
    
    def _calculer_charge_jour_enseignant(self, id_enseignant: int, jour: int, schedule: Dict) -> int:
        """Calcule la charge de l'enseignant pour un jour donné"""
        count = 0
        for seance in schedule.values():
            if (seance.get('id_enseignant') == id_enseignant and 
                seance.get('jour') == jour):
                count += 1
        return count
    
    def _calculer_charge_semaine_enseignant(self, id_enseignant: int, schedule: Dict) -> int:
        """Calcule la charge totale de l'enseignant pour la semaine"""
        count = 0
        for seance in schedule.values():
            if seance.get('id_enseignant') == id_enseignant:
                count += 1
        return count
    
    def _calculer_utilisation_salle_jour(self, codloc: str, jour: int, schedule: Dict) -> float:
        """Calcule le taux d'utilisation de la salle pour un jour donné"""
        count = 0
        for seance in schedule.values():
            if (seance.get('codloc') == codloc and 
                seance.get('jour') == jour):
                count += 1
        return count / 6.0  # 6 tranches horaires par jour
    
    def _calculer_dispersion_geographique(self, codgroupe: int, codloc: str, 
                                        jour: int, schedule: Dict) -> float:
        """
        Calcule la dispersion géographique (simplifiée)
        Idéalement, les cours d'un même groupe devraient être proches géographiquement
        """
        # Implémentation simplifiée - à améliorer avec vraies distances
        salles_groupe_jour = []
        for seance in schedule.values():
            if (seance.get('codgroupe') == codgroupe and 
                seance.get('jour') == jour):
                salles_groupe_jour.append(seance.get('codloc'))
        
        # Score simple basé sur la diversité des bâtiments
        if not salles_groupe_jour:
            return 0.0
        
        # Extraction des préfixes des codes de locaux (supposés indiquer le bâtiment)
        batiments = set(loc[:1] if loc else 'X' for loc in salles_groupe_jour)
        return len(batiments) / max(len(salles_groupe_jour), 1)
    
    def _evaluer_sequence_pedagogique(self, seance: Seance, jour: int, schedule: Dict) -> float:
        """
        Évalue si la séance respecte la séquence pédagogique logique
        Cours -> TD -> TP
        """
        seances_module = []
        for s in schedule.values():
            if (s.get('codform') == seance.codform and 
                s.get('codgroupe') == seance.codgroupe):
                seances_module.append({
                    'type': s.get('type_seance'),
                    'jour': s.get('jour'),
                    'tranche': s.get('tranche')
                })
        
        if not seances_module:
            return 1.0  # Pas d'autre séance, score neutre
        
        # Tri par ordre chronologique
        seances_module.sort(key=lambda x: (x['jour'], x['tranche']))
        
        # Vérification de la séquence logique
        types_order = [TypeSeance.COURS, TypeSeance.TD, TypeSeance.TP]
        score = 1.0
        
        for i in range(1, len(seances_module)):
            prev_type = seances_module[i-1]['type']
            curr_type = seances_module[i]['type']
            
            if (types_order.index(curr_type) < types_order.index(prev_type)):
                score -= 0.2  # Pénalité pour séquence inversée
        
        return max(0.0, score)
    
    def _calculer_espacement_seances_module(self, codform: str, jour: int, schedule: Dict) -> int:
        """
        Calcule l'espacement moyen entre les séances du même module
        """
        jours_seances = []
        for seance in schedule.values():
            if seance.get('codform') == codform:
                jours_seances.append(seance.get('jour'))
        
        if len(jours_seances) < 2:
            return 7  # Espacement maximal si pas assez de séances
        
        jours_seances.sort()
        espacements = [jours_seances[i+1] - jours_seances[i] 
                      for i in range(len(jours_seances)-1)]
        
        return int(np.mean(espacements)) if espacements else 7
    
    def fit_scaler(self, training_features: np.ndarray):
        """Ajuste le scaler sur les données d'entraînement"""
        self.scaler.fit(training_features)
        self.is_fitted = True
        logger.info("Scaler ajusté sur les données d'entraînement")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Normalise les caractéristiques"""
        if not self.is_fitted:
            logger.warning("Scaler non ajusté - utilisation des données brutes")
            return features
        
        return self.scaler.transform(features.reshape(1, -1))[0]

class ScoringNeuralNetwork:
    """Réseau de neurones pour le scoring de qualité des placements"""
    
    def __init__(self, input_dim: int = 18):
        self.input_dim = input_dim
        self.model = None
        self.feature_extractor = None
        self.history = None
    
    def build_model(self) -> Model:
        """Construit l'architecture du réseau de neurones"""
        
        # Input layer
        inputs = Input(shape=(self.input_dim,))
        
        # Première couche cachée avec normalisation
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Deuxième couche cachée
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Troisième couche cachée
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Couche de sortie - score entre 0 et 1
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compilation du modèle
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info(f"Modèle construit avec {self.input_dim} caractéristiques d'entrée")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Entraîne le modèle de scoring
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks pour l'entraînement
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Données de validation
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Entraînement
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation finale
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        results = {
            'train_loss': train_loss[0],
            'train_mae': train_loss[1],
            'epochs_trained': len(self.history.history['loss'])
        }
        
        if validation_data:
            val_loss = self.model.evaluate(X_val, y_val, verbose=0)
            results.update({
                'val_loss': val_loss[0],
                'val_mae': val_loss[1]
            })
        
        logger.info(f"Entraînement terminé - Loss: {results['train_loss']:.4f}, "
                   f"MAE: {results['train_mae']:.4f}")
        
        return results
    
    def predict_score(self, features: np.ndarray) -> float:
        """
        Prédit le score de qualité pour un placement
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        # Assurer que les features sont au bon format
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        score = self.model.predict(features, verbose=0)[0][0]
        return float(score)
    
    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """
        Prédit les scores pour un batch de placements
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        scores = self.model.predict(features_batch, verbose=0)
        return scores.flatten()
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        self.model.save(filepath)
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str):
        """Charge un modèle pré-entraîné"""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Modèle chargé: {filepath}")
    
    def get_model_summary(self) -> str:
        """Retourne un résumé du modèle"""
        if self.model is None:
            return "Aucun modèle construit"
        
        summary_str = []
        self.model.summary(print_fn=lambda x: summary_str.append(x))
        return '\n'.join(summary_str)

class TrainingDataGenerator:
    """Générateur de données d'entraînement pour le réseau de scoring"""
    
    def __init__(self, data_manager: DataManager, feature_extractor: FeatureExtractor):
        self.data_manager = data_manager
        self.feature_extractor = feature_extractor
    
    def generate_synthetic_data(self, nb_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère des données d'entraînement synthétiques
        """
        logger.info(f"Génération de {nb_samples} échantillons d'entraînement")
        
        features_list = []
        scores_list = []
        
        # Récupérer des sections existantes
        sections = list(self.data_manager.sections.keys())[:5]  # Limiter pour l'exemple
        
        for _ in range(nb_samples):
            # Sélection aléatoire d'une section
            codsec = np.random.choice(sections)
            
            # Génération d'une séance aléatoire
            seance = self._generate_random_seance(codsec)
            
            # Sélection d'un créneau aléatoire
            jour = np.random.randint(1, 7)
            tranche = np.random.randint(1, 7)
            
            # Sélection d'un local aléatoire adapté
            locaux_adaptes = [
                codloc for codloc, local in self.data_manager.locaux.items()
                if self.data_manager.est_local_adapte(codloc, seance.type_seance, 30)
            ]
            
            if not locaux_adaptes:
                continue
            
            codloc = np.random.choice(locaux_adaptes)
            
            # Sélection d'un enseignant compétent
            enseignants_competents = self.data_manager.trouver_enseignant_competent(
                seance.codform, seance.type_seance
            )
            
            if not enseignants_competents:
                id_enseignant = list(self.data_manager.enseignants.keys())[0]  # Fallback
            else:
                id_enseignant = np.random.choice(enseignants_competents)
            
            # Extraction des caractéristiques
            schedule_context = self._generate_random_schedule_context()
            features = self.feature_extractor.extract_features(
                seance, jour, tranche, codloc, id_enseignant, schedule_context
            )
            
            # Calcul du score de qualité
            score = self._calculate_quality_score(
                seance, jour, tranche, codloc, id_enseignant, schedule_context
            )
            
            features_list.append(features)
            scores_list.append(score)
        
        logger.info(f"Données générées: {len(features_list)} échantillons")
        return np.array(features_list), np.array(scores_list)
    
    def _generate_random_seance(self, codsec: str) -> Seance:
        """Génère une séance aléatoire pour une section"""
        # Récupérer les modules de la filière (simplifié)
        modules = list(self.data_manager.modules.keys())[:10]  # Limiter
        codform = np.random.choice(modules)
        
        type_seance = np.random.choice(list(TypeSeance))
        
        # Récupérer un groupe de la section
        groupes = [g for g in self.data_manager.groupes.values() if g.codsec == codsec]
        if groupes:
            codgroupe = groupes[0].codgroupe
        else:
            codgroupe = 1  # Fallback
        
        return Seance(
            codform=codform,
            type_seance=type_seance,
            codgroupe=codgroupe,
            duree=1.5
        )
    
    def _generate_random_schedule_context(self) -> Dict:
        """Génère un contexte d'emploi du temps aléatoire"""
        context = {}
        
        # Ajouter quelques séances aléatoirement placées
        for i in range(np.random.randint(0, 20)):
            context[f"seance_{i}"] = {
                'codform': f"Module_{np.random.randint(1, 10)}",
                'jour': np.random.randint(1, 7),
                'tranche': np.random.randint(1, 7),
                'codloc': f"S{np.random.randint(1, 50):02d}",
                'codgroupe': np.random.randint(1, 20),
                'id_enseignant': np.random.randint(1, 50),
                'type_seance': np.random.choice(['Cours', 'TD', 'TP'])
            }
        
        return context
    
    def _calculate_quality_score(self, seance: Seance, jour: int, tranche: int,
                               codloc: str, id_enseignant: int, 
                               schedule_context: Dict) -> float:
        """
        Calcule un score de qualité basé sur des règles heuristiques
        """
        score = 1.0
        
        # Pénalités basées sur l'horaire
        if tranche == 6:  # Dernière tranche moins désirable
            score -= 0.2
        
        if tranche == 1:  # Première tranche bonus
            score += 0.1
        
        if jour == 6:  # Jeudi moins désirable
            score -= 0.1
        
        # Vérification de la capacité de la salle
        local = self.data_manager.locaux.get(codloc)
        if local:
            effectif = calculer_effectif_groupe(self.data_manager, seance.codgroupe)
            ratio = local.capacite / max(effectif, 1)
            
            if ratio < 1.0:  # Salle trop petite
                score -= 0.5
            elif 1.0 <= ratio <= 2.0:  # Taille optimale
                score += 0.2
            elif ratio > 3.0:  # Salle trop grande
                score -= 0.1
        
        # Évaluation de la charge journalière
        charge_jour = sum(1 for s in schedule_context.values() 
                         if s.get('jour') == jour and s.get('codgroupe') == seance.codgroupe)
        
        if charge_jour > 6:  # Trop de cours dans la journée
            score -= 0.3
        elif 3 <= charge_jour <= 5:  # Charge optimale
            score += 0.1
        
        # Cohérence du type de local
        type_mapping = {
            TypeSeance.COURS: [TypeLocal.AMPHI, TypeLocal.SALLE],
            TypeSeance.TD: [TypeLocal.TD, TypeLocal.SALLE],
            TypeSeance.TP: [TypeLocal.TP, TypeLocal.SALLE]
        }
        
        if local and local.type in type_mapping.get(seance.type_seance, []):
            score += 0.15
        
        # Assurer que le score reste dans [0, 1]
        return max(0.0, min(1.0, score))

# Fonction principale pour l'entraînement
def train_scoring_model(data_manager: DataManager, 
                       output_dir: str = "models/") -> ScoringNeuralNetwork:
    """
    Fonction principale pour entraîner le modèle de scoring
    """
    logger.info("=== Début de l'entraînement du modèle de scoring ===")
    
    # Initialisation
    feature_extractor = FeatureExtractor(data_manager)
    training_generator = TrainingDataGenerator(data_manager, feature_extractor)
    
    # Génération des données d'entraînement
    X, y = training_generator.generate_synthetic_data(nb_samples=15000)
    
    # Ajustement du scaler
    feature_extractor.fit_scaler(X)
    
    # Normalisation des données
    X_normalized = feature_extractor.scaler.transform(X)
    
    # Division train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Données préparées - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Création et entraînement du modèle
    model = ScoringNeuralNetwork(input_dim=X.shape[1])
    results = model.train(X_train, y_train, X_val, y_val, epochs=150)
    
    # Sauvegarde
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_model(f"{output_dir}/scoring_model.h5")
    joblib.dump(feature_extractor.scaler, f"{output_dir}/feature_scaler.pkl")
    
    # Sauvegarde des métadonnées
    metadata = {
        'input_dim': X.shape[1],
        'training_samples': len(X),
        'training_results': results,
        'model_version': '1.0'
    }
    
    with open(f"{output_dir}/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=== Entraînement terminé avec succès ===")
    return model

if __name__ == "__main__":
    # Exemple d'utilisation
    from timetable_config import DataManager
    
    # Initialisation avec données simulées
    data_manager = DataManager()
    
    # Lancement de l'entraînement
    model = train_scoring_model(data_manager)
    
    print("Résumé du modèle:")
    print(model.get_model_summary())