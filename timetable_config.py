# timetable_config.py
"""
Configuration et structures de base pour l'algorithme IA d'affectation d'emplois du temps
Système développé pour l'USTHB - ENT
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypeSeance(Enum):
    COURS = "Cours"
    TD = "TD"
    TP = "TP"

class TypeLocal(Enum):
    AMPHI = "A"
    SALLE = "S"
    TD = "T"
    TP = "D"

class ConflictType(Enum):
    OK = 0
    CONFLIT_SALLE = 1
    CONFLIT_ENSEIGNANT = 2
    CONFLIT_HORAIRE = 3
    SURCHARGE = 4

@dataclass
class ConfigurationSysteme:
    """Configuration générale du système"""
    # Paramètres temporels
    JOURS_SEMAINE = 6  # Samedi à Jeudi
    TRANCHES_JOUR = 6  # 6 créneaux par jour
    
    # Contraintes de charge
    MAX_HEURES_JOUR_ETUDIANT = 9  # Maximum 6 créneaux = 9h
    MAX_HEURES_JOUR_ENSEIGNANT = 6  # Maximum 4 créneaux = 6h
    MAX_HEURES_SEMAINE_ENSEIGNANT = 20
    
    # Paramètres IA
    TAILLE_POPULATION_GENETIQUE = 50
    GENERATIONS_MAX = 100
    SEUIL_SATISFACTION = 0.95
    
    # Pondérations pour l'évaluation
    POIDS_CONTRAINTES_DURES = 0.4
    POIDS_EQUILIBRAGE_CHARGE = 0.2
    POIDS_UTILISATION_SALLES = 0.15
    POIDS_SATISFACTION_ENSEIGNANT = 0.15
    POIDS_SATISFACTION_ETUDIANT = 0.1

@dataclass
class Enseignant:
    """Structure pour représenter un enseignant"""
    id_enseignant: int
    nom: str
    prenom: str
    email: str
    grade: str
    # Préférences (à apprendre par IA)
    preference_matin: float = 0.5  # 0-1
    preference_aprem: float = 0.5  # 0-1
    charge_max: int = 20  # heures/semaine
    modules_competents: List[str] = None
    
    def __post_init__(self):
        if self.modules_competents is None:
            self.modules_competents = []

@dataclass
class Local:
    """Structure pour représenter un local"""
    codloc: str
    type: TypeLocal
    capacite: int
    faculte: Optional[str] = None
    equipements: List[str] = None
    
    def __post_init__(self):
        if self.equipements is None:
            self.equipements = []

@dataclass
class Module:
    """Structure pour représenter un module"""
    codform: str
    titre: str
    codfil: str
    nbr_cours: int
    nbr_tds: int
    nbr_tps: int
    type_module: str = "Présentiel"

@dataclass
class Section:
    """Structure pour représenter une section"""
    codsec: str
    section: str
    codfil: str
    annee: int
    effectif: int = 30  # effectif moyen par défaut

@dataclass
class Groupe:
    """Structure pour représenter un groupe"""
    codgroupe: int
    codsec: str
    numero_groupe: int
    effectif: int = 30

@dataclass
class Seance:
    """Structure pour représenter une séance"""
    codform: str
    type_seance: TypeSeance
    codgroupe: int
    duree: float = 1.5  # en heures
    enseignant_requis: Optional[int] = None
    local_requis: Optional[str] = None
    
    # Position dans l'EDT (à déterminer)
    jour: Optional[int] = None
    tranche: Optional[int] = None
    codloc: Optional[str] = None
    id_enseignant: Optional[int] = None

@dataclass
class CreneauHoraire:
    """Structure pour représenter un créneau horaire"""
    jour: int  # 1-6 (Samedi à Jeudi)
    tranche: int  # 1-6
    
    def __str__(self):
        jours = ["", "Samedi", "Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi"]
        tranches = ["", "08:00-09:30", "09:40-11:10", "11:20-12:50", 
                   "13:00-14:30", "14:40-16:10", "16:20-17:50"]
        return f"{jours[self.jour]} {tranches[self.tranche]}"

class DataManager:
    """Gestionnaire de données pour l'interface avec la base de données"""
    
    def __init__(self):
        self.enseignants: Dict[int, Enseignant] = {}
        self.locaux: Dict[str, Local] = {}
        self.modules: Dict[str, Module] = {}
        self.sections: Dict[str, Section] = {}
        self.groupes: Dict[int, Groupe] = {}
        self.config = ConfigurationSysteme()
    
    def charger_donnees_mysql(self, connexion_db):
        """
        Charge les données depuis la base MySQL
        connexion_db: connexion à la base de données MySQL
        """
        try:
            # Chargement des enseignants
            query_enseignants = """
            SELECT id_enseignant, nom, prenom, email, telephone, grade 
            FROM enseignants
            """
            df_enseignants = pd.read_sql(query_enseignants, connexion_db)
            
            for _, row in df_enseignants.iterrows():
                self.enseignants[row['id_enseignant']] = Enseignant(
                    id_enseignant=row['id_enseignant'],
                    nom=row['nom'],
                    prenom=row['prenom'],
                    email=row['email'],
                    grade=row['grade']
                )
            
            # Chargement des locaux
            query_locaux = """
            SELECT codloc, type, capacite, faculte 
            FROM local
            """
            df_locaux = pd.read_sql(query_locaux, connexion_db)
            
            for _, row in df_locaux.iterrows():
                self.locaux[row['codloc']] = Local(
                    codloc=row['codloc'],
                    type=TypeLocal(row['type']),
                    capacite=row['capacite'],
                    faculte=row['faculte']
                )
            
            # Chargement des modules
            query_modules = """
            SELECT codform, titre_form, codfil, nbr_cours, nbr_tds, nbr_tps, type 
            FROM offre_formation
            """
            df_modules = pd.read_sql(query_modules, connexion_db)
            
            for _, row in df_modules.iterrows():
                self.modules[row['codform']] = Module(
                    codform=row['codform'],
                    titre=row['titre_form'],
                    codfil=row['codfil'],
                    nbr_cours=row['nbr_cours'] or 0,
                    nbr_tds=row['nbr_tds'] or 0,
                    nbr_tps=row['nbr_tps'] or 0,
                    type_module=row['type'] or "Présentiel"
                )
            
            # Chargement des sections
            query_sections = """
            SELECT codsec, section, codfil, annee 
            FROM sections
            """
            df_sections = pd.read_sql(query_sections, connexion_db)
            
            for _, row in df_sections.iterrows():
                self.sections[row['codsec']] = Section(
                    codsec=row['codsec'],
                    section=row['section'],
                    codfil=row['codfil'],
                    annee=row['annee']
                )
            
            # Chargement des groupes
            query_groupes = """
            SELECT codgroupe, codsec, groupe 
            FROM groupes
            """
            df_groupes = pd.read_sql(query_groupes, connexion_db)
            
            for _, row in df_groupes.iterrows():
                self.groupes[row['codgroupe']] = Groupe(
                    codgroupe=row['codgroupe'],
                    codsec=row['codsec'],
                    numero_groupe=row['groupe']
                )
            
            # Chargement des compétences enseignant-module
            self._charger_competences_enseignants(connexion_db)
            
            logger.info(f"Données chargées: {len(self.enseignants)} enseignants, "
                       f"{len(self.locaux)} locaux, {len(self.modules)} modules, "
                       f"{len(self.sections)} sections, {len(self.groupes)} groupes")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
    
    def _charger_competences_enseignants(self, connexion_db):
        """Charge les compétences des enseignants pour chaque module"""
        query_competences = """
        SELECT id_enseignant, codform, type_seance 
        FROM enseignant_module
        """
        df_competences = pd.read_sql(query_competences, connexion_db)
        
        for _, row in df_competences.iterrows():
            id_ens = row['id_enseignant']
            if id_ens in self.enseignants:
                module_type = f"{row['codform']}_{row['type_seance']}"
                self.enseignants[id_ens].modules_competents.append(module_type)
    
    def generer_seances_requises(self, codsec: str) -> List[Seance]:
        """
        Génère la liste des séances requises pour une section donnée
        """
        seances = []
        
        if codsec not in self.sections:
            logger.warning(f"Section {codsec} non trouvée")
            return seances
        
        # Récupération des modules de la section
        # (Ici, nous devrons faire une requête pour modules_section)
        # Pour l'exemple, on suppose une méthode qui retourne les modules
        modules_section = self._get_modules_section(codsec)
        
        # Récupération des groupes de la section
        groupes_section = [g for g in self.groupes.values() if g.codsec == codsec]
        
        for codform in modules_section:
            if codform not in self.modules:
                continue
                
            module = self.modules[codform]
            
            # Génération des séances de cours
            for _ in range(module.nbr_cours):
                # Les cours se font pour toute la section (tous les groupes ensemble)
                if groupes_section:
                    seances.append(Seance(
                        codform=codform,
                        type_seance=TypeSeance.COURS,
                        codgroupe=groupes_section[0].codgroupe,  # Représentant de la section
                        duree=1.5
                    ))
            
            # Génération des séances de TD
            for _ in range(module.nbr_tds):
                for groupe in groupes_section:
                    seances.append(Seance(
                        codform=codform,
                        type_seance=TypeSeance.TD,
                        codgroupe=groupe.codgroupe,
                        duree=1.5
                    ))
            
            # Génération des séances de TP
            for _ in range(module.nbr_tps):
                for groupe in groupes_section:
                    seances.append(Seance(
                        codform=codform,
                        type_seance=TypeSeance.TP,
                        codgroupe=groupe.codgroupe,
                        duree=1.5
                    ))
        
        logger.info(f"Génération de {len(seances)} séances pour la section {codsec}")
        return seances
    
    def _get_modules_section(self, codsec: str) -> List[str]:
        """
        Récupère les modules d'une section depuis la base de données
        Cette méthode devrait faire une requête SQL réelle
        """
        # Simulation - à remplacer par une vraie requête
        return ["Alg-1", "Ana-1", "Algo-1"]  # Exemple
    
    def trouver_enseignant_competent(self, codform: str, type_seance: TypeSeance) -> List[int]:
        """
        Trouve les enseignants compétents pour un module et type de séance donnés
        """
        module_type = f"{codform}_{type_seance.value}"
        competents = []
        
        for id_ens, enseignant in self.enseignants.items():
            if module_type in enseignant.modules_competents:
                competents.append(id_ens)
        
        return competents
    
    def est_local_adapte(self, codloc: str, type_seance: TypeSeance, effectif: int) -> bool:
        """
        Vérifie si un local est adapté pour un type de séance et un effectif donnés
        """
        if codloc not in self.locaux:
            return False
        
        local = self.locaux[codloc]
        
        # Vérification de la capacité
        if local.capacite < effectif:
            return False
        
        # Vérification du type de local vs type de séance
        if type_seance == TypeSeance.COURS:
            return local.type in [TypeLocal.AMPHI, TypeLocal.SALLE]
        elif type_seance == TypeSeance.TD:
            return local.type in [TypeLocal.TD, TypeLocal.SALLE]
        elif type_seance == TypeSeance.TP:
            return local.type in [TypeLocal.TP, TypeLocal.SALLE]
        
        return True
    
    def sauvegarder_edt(self, solution: Dict, connexion_db):
        """
        Sauvegarde l'emploi du temps généré dans la base de données
        """
        try:
            # Vider les anciennes séances (optionnel)
            # connexion_db.execute("DELETE FROM seances_edt WHERE ...")
            
            for section_schedule in solution.values():
                for seance_data in section_schedule.values():
                    query = """
                    INSERT INTO seances_edt 
                    (codform, codloc, jour, tranche, type_seance, codgroupe, id_enseignant)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cursor = connexion_db.cursor()
                    cursor.execute(query, (
                        seance_data['codform'],
                        seance_data['codloc'],
                        seance_data['jour'],
                        seance_data['tranche'],
                        seance_data['type_seance'],
                        seance_data['codgroupe'],
                        seance_data['id_enseignant']
                    ))
            
            connexion_db.commit()
            logger.info("Emploi du temps sauvegardé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            connexion_db.rollback()
            raise

# Utilitaires
def creer_matrice_contraintes(data_manager: DataManager) -> np.ndarray:
    """
    Crée une matrice de contraintes pour optimiser les calculs
    """
    nb_jours = data_manager.config.JOURS_SEMAINE
    nb_tranches = data_manager.config.TRANCHES_JOUR
    nb_locaux = len(data_manager.locaux)
    
    # Matrice 3D: [jour][tranche][local] = disponibilité (1=libre, 0=occupé)
    matrice = np.ones((nb_jours + 1, nb_tranches + 1, nb_locaux), dtype=int)
    
    return matrice

def calculer_effectif_groupe(data_manager: DataManager, codgroupe: int) -> int:
    """
    Calcule l'effectif d'un groupe
    """
    if codgroupe in data_manager.groupes:
        return data_manager.groupes[codgroupe].effectif
    return 30  # effectif par défaut