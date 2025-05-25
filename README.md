# IA
Algorithme IA pour Génération Automatique d'Emplois du Temps - USTHB
1. Architecture du Système IA
1.1 Approche Hybride Proposée

Optimisation par Contraintes (Constraint Satisfaction Problem - CSP)
Réseau de Neurones pour l'apprentissage des préférences
Algorithme Génétique pour l'optimisation fine
Système Expert pour les règles métier

1.2 Modèles IA Utilisés
A. Réseau de Neurones de Scoring (RNS)
Input Layer: 
- Caractéristiques de la séance (type, durée, groupe_size)
- Caractéristiques du local (capacité, type, équipements)
- Contexte temporel (jour, tranche, adjacent_courses)
- Profil enseignant (préférences, charge)

Hidden Layers:
- Layer 1: 128 neurones (ReLU)
- Layer 2: 64 neurones (ReLU)
- Layer 3: 32 neurones (ReLU)

Output Layer: Score de qualité [0,1]
B. Réseau de Neurones de Classification (RNC)
Objectif: Classifier les conflits potentiels
Classes: [OK, CONFLIT_SALLE, CONFLIT_ENSEIGNANT, CONFLIT_HORAIRE, SURCHARGE]
2. Algorithme Principal
2.1 Phase 1: Préparation des Données
pythondef prepare_data():
    # Extraction depuis la BDD
    sections = get_sections()
    modules = get_modules_by_section()
    enseignants = get_enseignants_with_constraints()
    locaux = get_locaux_with_capacity()
    
    # Calcul des besoins
    seances_required = calculate_required_sessions()
    
    # Matrice de contraintes
    constraint_matrix = build_constraint_matrix()
    
    return {
        'sections': sections,
        'modules': modules,
        'enseignants': enseignants,
        'locaux': locaux,
        'seances_required': seances_required,
        'constraints': constraint_matrix
    }
2.2 Phase 2: Génération par IA
pythonclass TimetableAI:
    def __init__(self):
        self.scoring_model = load_neural_network_model('scoring_nn.h5')
        self.conflict_classifier = load_neural_network_model('conflict_classifier.h5')
        self.genetic_optimizer = GeneticAlgorithm()
        
    def generate_timetable(self, data):
        # Étape 1: Génération initiale par CSP
        initial_solution = self.csp_initial_generation(data)
        
        # Étape 2: Amélioration par IA
        optimized_solution = self.ai_optimization(initial_solution, data)
        
        # Étape 3: Validation finale
        validated_solution = self.validate_and_repair(optimized_solution, data)
        
        return validated_solution
    
    def csp_initial_generation(self, data):
        """Génération initiale basée sur les contraintes dures"""
        solution = {}
        
        for section in data['sections']:
            section_schedule = {}
            
            # Pour chaque module de la section
            for module in data['modules'][section['codsec']]:
                # Générer les séances requises
                required_sessions = self.calculate_module_sessions(module)
                
                for session in required_sessions:
                    # Trouver le meilleur créneau avec IA
                    best_slot = self.find_best_slot_ai(session, section_schedule, data)
                    
                    if best_slot:
                        section_schedule[best_slot['id']] = {
                            'codform': module['codform'],
                            'type_seance': session['type'],
                            'codgroupe': session['groupe'],
                            'codloc': best_slot['local'],
                            'jour': best_slot['jour'],
                            'tranche': best_slot['tranche'],
                            'id_enseignant': session['enseignant']
                        }
            
            solution[section['codsec']] = section_schedule
        
        return solution
    
    def find_best_slot_ai(self, session, current_schedule, data):
        """Utilise l'IA pour trouver le meilleur créneau"""
        candidates = []
        
        # Générer tous les créneaux possibles
        for jour in range(1, 7):  # Samedi à Jeudi
            for tranche in range(1, 7):  # 6 tranches horaires
                for local in data['locaux']:
                    # Vérification des contraintes dures
                    if self.check_hard_constraints(session, jour, tranche, local, current_schedule, data):
                        
                        # Calcul du score IA
                        features = self.extract_features(session, jour, tranche, local, current_schedule, data)
                        ai_score = self.scoring_model.predict([features])[0][0]
                        
                        # Vérification des conflits avec IA
                        conflict_prob = self.conflict_classifier.predict([features])[0]
                        conflict_class = np.argmax(conflict_prob)
                        
                        if conflict_class == 0:  # Pas de conflit
                            candidates.append({
                                'id': f"{jour}_{tranche}_{local['codloc']}",
                                'jour': jour,
                                'tranche': tranche,
                                'local': local['codloc'],
                                'ai_score': ai_score,
                                'conflict_score': conflict_prob[0]
                            })
        
        # Retourner le meilleur candidat
        if candidates:
            return max(candidates, key=lambda x: x['ai_score'] * x['conflict_score'])
        
        return None
    
    def extract_features(self, session, jour, tranche, local, current_schedule, data):
        """Extraction des caractéristiques pour l'IA"""
        features = []
        
        # Caractéristiques de la séance
        features.extend([
            1 if session['type'] == 'Cours' else 0,
            1 if session['type'] == 'TD' else 0,
            1 if session['type'] == 'TP' else 0,
            session.get('duree', 1.5),
            session.get('groupe_size', 20)
        ])
        
        # Caractéristiques du local
        features.extend([
            local['capacite'],
            1 if local['type'] == 'A' else 0,  # Amphi
            1 if local['type'] == 'S' else 0,  # Salle
            1 if local['type'] == 'T' else 0,  # TD
            1 if local['type'] == 'D' else 0   # TP
        ])
        
        # Contexte temporel
        features.extend([
            jour / 6.0,  # Normalisation
            tranche / 6.0,
            self.count_adjacent_courses(jour, tranche, current_schedule),
            self.calculate_day_load(jour, current_schedule)
        ])
        
        # Profil enseignant
        if session.get('enseignant'):
            enseignant = self.get_enseignant_profile(session['enseignant'], data)
            features.extend([
                enseignant.get('preference_matin', 0.5),
                enseignant.get('preference_aprem', 0.5),
                enseignant.get('charge_actuelle', 0) / 20.0  # Normalisation
            ])
        else:
            features.extend([0.5, 0.5, 0])
        
        # Contraintes de répartition
        features.extend([
            self.calculate_module_distribution(session['codform'], jour, current_schedule),
            self.calculate_group_load(session['groupe'], jour, current_schedule)
        ])
        
        return np.array(features)
    
    def ai_optimization(self, initial_solution, data):
        """Optimisation par algorithme génétique avec scoring IA"""
        population = self.genetic_optimizer.create_population(initial_solution, size=50)
        
        for generation in range(100):
            # Évaluation avec scoring IA
            fitness_scores = []
            for individual in population:
                score = self.evaluate_solution_ai(individual, data)
                fitness_scores.append(score)
            
            # Sélection, croisement, mutation
            population = self.genetic_optimizer.evolve(population, fitness_scores)
            
            # Critère d'arrêt
            best_score = max(fitness_scores)
            if best_score > 0.95:  # Score de satisfaction élevé
                break
        
        # Retourner la meilleure solution
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]
    
    def evaluate_solution_ai(self, solution, data):
        """Évaluation globale d'une solution avec IA"""
        total_score = 0
        total_sessions = 0
        
        for section_schedule in solution.values():
            for session in section_schedule.values():
                # Score IA pour chaque séance
                features = self.extract_features_from_solution(session, solution, data)
                session_score = self.scoring_model.predict([features])[0][0]
                
                total_score += session_score
                total_sessions += 1
        
        return total_score / total_sessions if total_sessions > 0 else 0
3. Entraînement des Modèles IA
3.1 Génération des Données d'Entraînement
pythondef generate_training_data():
    """Génère des données d'entraînement à partir des emplois du temps existants"""
    
    # Extraction des emplois du temps historiques
    historical_schedules = extract_historical_data()
    
    training_features = []
    training_labels_score = []
    training_labels_conflict = []
    
    for schedule in historical_schedules:
        for session in schedule['sessions']:
            # Extraction des caractéristiques
            features = extract_session_features(session, schedule)
            
            # Labels pour le scoring (qualité subjective de 0 à 1)
            quality_score = calculate_quality_score(session, schedule)
            
            # Labels pour la classification des conflits
            conflict_type = detect_conflict_type(session, schedule)
            
            training_features.append(features)
            training_labels_score.append(quality_score)
            training_labels_conflict.append(conflict_type)
    
    return {
        'features': np.array(training_features),
        'scores': np.array(training_labels_score),
        'conflicts': np.array(training_labels_conflict)
    }

def calculate_quality_score(session, schedule):
    """Calcule un score de qualité basé sur des critères objectifs"""
    score = 1.0
    
    # Pénalités
    if session['tranche'] == 6:  # Dernière tranche
        score -= 0.2
    
    if session['jour'] == 6:  # Jeudi
        score -= 0.1
    
    # Bonus pour répartition équilibrée
    daily_load = count_daily_sessions(session['jour'], schedule)
    if 2 <= daily_load <= 4:
        score += 0.1
    
    # Adéquation salle/type de cours
    if is_appropriate_room(session['codloc'], session['type_seance']):
        score += 0.2
    
    return max(0, min(1, score))
3.2 Architecture des Réseaux de Neurones
pythondef create_scoring_model():
    """Crée le modèle de scoring"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(feature_count,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Score entre 0 et 1
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_conflict_classifier():
    """Crée le modèle de classification des conflits"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(feature_count,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')  # 5 classes de conflits
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
4. Intégration avec le Backend PHP
4.1 API PHP pour l'IA
php<?php
class TimetableAIController {
    
    public function generateWithAI() {
        try {
            // Préparation des données
            $data = $this->prepareDataForAI();
            
            // Appel du service IA (Python)
            $aiResult = $this->callAIService($data);
            
            // Validation et sauvegarde
            if ($this->validateAIResult($aiResult)) {
                $this->saveGeneratedTimetable($aiResult);
                return json_encode(['success' => true, 'data' => $aiResult]);
            }
            
        } catch (Exception $e) {
            return json_encode(['success' => false, 'error' => $e->getMessage()]);
        }
    }
    
    private function callAIService($data) {
        // Appel du service Python via API REST ou subprocess
        $python_script = "python3 /path/to/timetable_ai.py";
        $json_data = json_encode($data);
        
        $command = "echo '$json_data' | $python_script";
        $result = shell_exec($command);
        
        return json_decode($result, true);
    }
    
    private function prepareDataForAI() {
        $pdo = Database::getConnection();
        
        // Extraction des données nécessaires
        $sections = $this->getSections($pdo);
        $modules = $this->getModules($pdo);
        $enseignants = $this->getEnseignants($pdo);
        $locaux = $this->getLocaux($pdo);
        
        return [
            'sections' => $sections,
            'modules' => $modules,
            'enseignants' => $enseignants,
            'locaux' => $locaux,
            'constraints' => $this->getConstraints($pdo)
        ];
    }
}
?>
5. Interface Frontend JavaScript
5.1 Interface de Génération IA
javascriptclass TimetableAIGenerator {
    constructor() {
        this.apiUrl = '/api/timetable';
        this.isGenerating = false;
    }
    
    async generateWithAI(sectionCode = null) {
        if (this.isGenerating) return;
        
        this.isGenerating = true;
        this.showProgressBar();
        
        try {
            const response = await fetch(`${this.apiUrl}/generate-ai`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    section: sectionCode,
                    preferences: this.getUserPreferences()
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayGeneratedTimetable(result.data);
                this.showSuccessMessage('Emploi du temps généré avec succès par IA!');
            } else {
                this.showErrorMessage(result.error);
            }
            
        } catch (error) {
            this.showErrorMessage('Erreur lors de la génération: ' + error.message);
        } finally {
            this.isGenerating = false;
            this.hideProgressBar();
        }
    }
    
    getUserPreferences() {
        return {
            prefer_morning: document.getElementById('prefer_morning').checked,
            avoid_thursday: document.getElementById('avoid_thursday').checked,
            max_daily_hours: parseInt(document.getElementById('max_daily_hours').value),
            preferred_rooms: this.getSelectedRooms()
        };
    }
    
    displayGeneratedTimetable(timetableData) {
        const container = document.getElementById('timetable-container');
        container.innerHTML = '';
        
        // Affichage du planning généré
        for (const [sectionCode, schedule] of Object.entries(timetableData)) {
            const sectionDiv = this.createSectionDisplay(sectionCode, schedule);
            container.appendChild(sectionDiv);
        }
        
        // Options de validation/modification
        this.addValidationControls(container);
    }
    
    addValidationControls(container) {
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'ai-controls';
        controlsDiv.innerHTML = `
            <button id="validate-ai" class="btn btn-success">Valider l'emploi du temps</button>
            <button id="refine-ai" class="btn btn-warning">Affiner avec IA</button>
            <button id="manual-edit" class="btn btn-info">Modification manuelle</button>
        `;
        
        container.appendChild(controlsDiv);
        
        // Event listeners
        document.getElementById('validate-ai').addEventListener('click', () => {
            this.validateAndSave();
        });
        
        document.getElementById('refine-ai').addEventListener('click', () => {
            this.refineWithAI();
        });
    }
}

// Initialisation
const timetableAI = new TimetableAIGenerator();
6. Critères d'Évaluation et Métriques
6.1 Métriques de Qualité
pythondef evaluate_timetable_quality(solution, data):
    """Évalue la qualité globale d'un emploi du temps"""
    
    metrics = {
        'constraint_satisfaction': 0,
        'load_balancing': 0,
        'room_utilization': 0,
        'teacher_satisfaction': 0,
        'student_satisfaction': 0
    }
    
    # Satisfaction des contraintes dures
    metrics['constraint_satisfaction'] = check_hard_constraints_satisfaction(solution)
    
    # Équilibrage de la charge
    metrics['load_balancing'] = calculate_load_balance(solution)
    
    # Utilisation optimale des salles
    metrics['room_utilization'] = calculate_room_efficiency(solution, data)
    
    # Satisfaction des enseignants (préférences horaires)
    metrics['teacher_satisfaction'] = calculate_teacher_satisfaction(solution, data)
    
    # Satisfaction des étudiants (répartition équilibrée)
    metrics['student_satisfaction'] = calculate_student_satisfaction(solution)
    
    # Score global pondéré
    weights = {
        'constraint_satisfaction': 0.4,
        'load_balancing': 0.2,
        'room_utilization': 0.15,
        'teacher_satisfaction': 0.15,
        'student_satisfaction': 0.1
    }
    
    global_score = sum(metrics[key] * weights[key] for key in metrics)
    
    return {
        'metrics': metrics,
        'global_score': global_score
    }
7. Déploiement et Maintenance
7.1 Pipeline de Déploiement

Entraînement initial des modèles sur données historiques
Tests de validation sur emplois du temps existants
Déploiement graduel (section par section)
Monitoring des performances
Réentraînement périodique avec nouvelles données

7.2 Amélioration Continue
pythondef continuous_learning_pipeline():
    """Pipeline d'apprentissage continu"""
    
    # Collecte des feedbacks
    feedbacks = collect_user_feedbacks()
    
    # Mise à jour des données d'entraînement
    new_training_data = update_training_data(feedbacks)
    
    # Réentraînement des modèles
    if len(new_training_data) > threshold:
        retrain_models(new_training_data)
        
    # Validation des nouveaux modèles
    validate_updated_models()
    
    # Déploiement automatique si validation OK
    deploy_if_improved()
Ce système hybride combine la robustesse des approches classiques (CSP) avec la flexibilité et l'apprentissage de l'IA, permettant une génération automatique d'emplois du temps de haute qualité tout en s'adaptant aux spécificités de votre établissement.