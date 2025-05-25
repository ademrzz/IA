

<?php
/**
 * Système d'intégration Backend PHP avec IA Python - VERSION COMPLÈTE
 * Étape 4 : Intégration avec le Backend PHP
 * USTHB - Génération automatique d'emplois du temps
 */

// ==================== CLASSE DE CONNEXION DATABASE ====================
class Database {
    private static $connection = null;
    
    public static function getConnection() {
        if (self::$connection === null) {
            try {
                self::$connection = new PDO(
                    "mysql:host=" . DB_HOST . ";dbname=" . DB_NAME . ";charset=utf8mb4",
                    DB_USER,
                    DB_PASS,
                    [
                        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                        PDO::ATTR_EMULATE_PREPARES => false
                    ]
                );
            } catch (PDOException $e) {
                error_log("Erreur connexion BDD: " . $e->getMessage());
                throw new Exception("Erreur de connexion à la base de données");
            }
        }
        return self::$connection;
    }
}

// ==================== CONTRÔLEUR PRINCIPAL IA ====================
class TimetableAIController {
    
    private $pdo;
    private $pythonScriptPath;
    private $modelsPath;
    private $tempPath;
    private $logPath;
    
    public function __construct() {
        $this->pdo = Database::getConnection();
        $this->pythonScriptPath = __DIR__ . '/../python/timetable_ai_service.py';
        $this->modelsPath = __DIR__ . '/../models/';
        $this->tempPath = __DIR__ . '/../temp/';
        $this->logPath = __DIR__ . '/../logs/';
        
        // Créer les dossiers s'ils n'existent pas
        $this->createDirectories();
    }
    
    private function createDirectories() {
        $dirs = [$this->tempPath, $this->logPath, $this->modelsPath];
        foreach ($dirs as $dir) {
            if (!is_dir($dir)) {
                mkdir($dir, 0755, true);
            }
        }
    }
    
    /**
     * Génère un emploi du temps avec IA pour une section donnée
     */
    public function generateTimetableAI($codsec, $options = []) {
        $startTime = microtime(true);
        $logId = $this->createGenerationLog($codsec, 'ai_generation');
        
        try {
            // 1. Validation des entrées
            if (!$this->validateSection($codsec)) {
                throw new Exception("Section invalide: $codsec");
            }
            
            // 2. Préparation des données
            $data = $this->prepareAIData($codsec, $options);
            $this->updateGenerationLog($logId, 'data_preparation_complete', $data['stats'] ?? []);
            
            // 3. Appel du service IA Python
            $aiResult = $this->callAIService($data);
            $this->updateGenerationLog($logId, 'ai_processing_complete', $aiResult['metrics'] ?? []);
            
            // 4. Validation du résultat
            if (!$this->validateAIResult($aiResult)) {
                throw new Exception("Résultat IA invalide");
            }
            
            // 5. Sauvegarde en base de données
            $timetableId = $this->saveTimetableFromAI($aiResult, $codsec);
            
            $endTime = microtime(true);
            $executionTime = round($endTime - $startTime, 2);
            
            // 6. Finalisation du log
            $this->updateGenerationLog($logId, 'completed', [
                'timetable_id' => $timetableId,
                'execution_time' => $executionTime
            ]);
            
            return [
                'success' => true,
                'timetable_id' => $timetableId,
                'section' => $codsec,
                'fitness_score' => $aiResult['fitness_final'],
                'nb_seances' => $aiResult['nb_seances'],
                'conflicts' => $aiResult['solution_summary']['conflicts'] ?? [],
                'generation_time' => $executionTime,
                'metrics' => $aiResult['solution_summary']['metrics'] ?? [],
                'log_id' => $logId
            ];
            
        } catch (Exception $e) {
            $this->updateGenerationLog($logId, 'error', ['error' => $e->getMessage()]);
            error_log("Erreur génération IA: " . $e->getMessage());
            
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'section' => $codsec,
                'log_id' => $logId
            ];
        }
    }
    
    /**
     * Génère des emplois du temps pour plusieurs sections
     */
    public function generateMultipleSections($sections, $options = []) {
        $results = [];
        $overallSuccess = true;
        $startTime = microtime(true);
        
        foreach ($sections as $codsec) {
            $result = $this->generateTimetableAI($codsec, $options);
            $results[$codsec] = $result;
            
            if (!$result['success']) {
                $overallSuccess = false;
            }
            
            // Petite pause entre les générations pour éviter la surcharge
            usleep(500000); // 0.5 seconde
        }
        
        $totalTime = round(microtime(true) - $startTime, 2);
        
        return [
            'success' => $overallSuccess,
            'results' => $results,
            'total_sections' => count($sections),
            'successful_sections' => count(array_filter($results, fn($r) => $r['success'])),
            'total_time' => $totalTime,
            'summary' => $this->generateBatchSummary($results)
        ];
    }
    
    /**
     * Prépare les données pour le service IA
     */
    private function prepareAIData($codsec, $options) {
        // Extraction des données de base
        $sections = $this->getSectionData($codsec);
        $modules = $this->getModulesData($codsec);
        $enseignants = $this->getEnseignantsData();
        $locaux = $this->getLocauxData();
        $groupes = $this->getGroupesData($codsec);
        $contraintes = $this->getContraintesData($codsec);
        
        // Calcul des statistiques
        $stats = [
            'nb_modules' => count($modules),
            'nb_enseignants' => count($enseignants),
            'nb_locaux' => count($locaux),
            'nb_groupes' => count($groupes)
        ];
        
        // Préférences utilisateur avec valeurs par défaut
        $preferences = array_merge([
            'prefer_morning' => true,
            'avoid_thursday' => false,
            'max_daily_hours_teacher' => 6,
            'max_daily_hours_student' => 5,
            'preferred_room_types' => [],
            'optimization_time_limit' => 300, // 5 minutes
            'genetic_population_size' => 50,
            'genetic_generations' => 100,
            'neural_threshold' => 0.7
        ], $options);
        
        return [
            'sections' => $sections,
            'modules' => $modules,
            'enseignants' => $enseignants,
            'locaux' => $locaux,
            'groupes' => $groupes,
            'contraintes' => $contraintes,
            'preferences' => $preferences,
            'config' => [
                'JOURS_SEMAINE' => 6,
                'TRANCHES_JOUR' => 6,
                'DUREE_SEANCE' => 1.5
            ],
            'stats' => $stats
        ];
    }
    
    /**
     * Appelle le service IA Python
     */
    private function callAIService($data) {
        // Sauvegarde des données dans un fichier temporaire
        $tempFile = $this->tempPath . 'ai_input_' . uniqid() . '.json';
        $outputFile = $this->tempPath . 'ai_output_' . uniqid() . '.json';
        
        file_put_contents($tempFile, json_encode($data, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
        
        // Construction de la commande Python
        $command = sprintf(
            "python3 %s --input %s --output %s --models %s 2>&1",
            escapeshellarg($this->pythonScriptPath),
            escapeshellarg($tempFile),
            escapeshellarg($outputFile),
            escapeshellarg($this->modelsPath)
        );
        
        // Execution avec timeout
        $startTime = time();
        $timeout = 600; // 10 minutes
        
        $process = popen($command, 'r');
        if (!$process) {
            throw new Exception("Impossible de lancer le processus Python");
        }
        
        $output = '';
        while (!feof($process) && (time() - $startTime) < $timeout) {
            $output .= fread($process, 4096);
        }
        
        $returnCode = pclose($process);
        
        // Vérification du timeout
        if ((time() - $startTime) >= $timeout) {
            throw new Exception("Timeout: Le processus IA a dépassé la limite de temps");
        }
        
        // Lecture du fichier de sortie
        if (!file_exists($outputFile)) {
            throw new Exception("Fichier de sortie IA non généré. Sortie: " . $output);
        }
        
        $result = json_decode(file_get_contents($outputFile), true);
        
        // Nettoyage des fichiers temporaires
        unlink($tempFile);
        unlink($outputFile);
        
        if ($returnCode !== 0) {
            throw new Exception("Erreur Python (code: $returnCode): " . $output);
        }
        
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new Exception("Réponse Python invalide: " . json_last_error_msg());
        }
        
        return $result;
    }
    
    /**
     * Sauvegarde l'emploi du temps généré par IA
     */
    private function saveTimetableFromAI($aiResult, $codsec) {
        $this->pdo->beginTransaction();
        
        try {
            // 1. Supprimer l'ancien emploi du temps s'il existe
            $this->deleteExistingTimetable($codsec);
            
            // 2. Créer l'enregistrement principal de l'emploi du temps
            $stmt = $this->pdo->prepare("
                INSERT INTO emplois_temps (codsec, statut, date_creation, type_generation, 
                                         fitness_score, nb_seances, metadata, version)
                VALUES (?, 'genere_ia', NOW(), 'ia_automatique', ?, ?, ?, 1)
            ");
            
            $metadata = json_encode([
                'ai_metrics' => $aiResult['solution_summary']['metrics'] ?? [],
                'conflicts' => $aiResult['solution_summary']['conflicts'] ?? [],
                'optimization_stats' => $aiResult['solution_summary']['optimization_stats'] ?? [],
                'generation_params' => $aiResult['generation_params'] ?? []
            ]);
            
            $stmt->execute([
                $codsec,
                $aiResult['fitness_final'],
                $aiResult['nb_seances'],
                $metadata
            ]);
            
            $timetableId = $this->pdo->lastInsertId();
            
            // 3. Sauvegarder les séances
            if (isset($aiResult['emploi_temps'])) {
                $this->saveSeancesFromAI($timetableId, $aiResult['emploi_temps'], $codsec);
            }
            
            // 4. Enregistrer les statistiques de génération
            $this->saveGenerationStats($timetableId, $aiResult);
            
            // 5. Créer un rapport de qualité
            $this->createQualityReport($timetableId, $aiResult);
            
            $this->pdo->commit();
            return $timetableId;
            
        } catch (Exception $e) {
            $this->pdo->rollback();
            throw new Exception("Erreur sauvegarde: " . $e->getMessage());
        }
    }
    
    /**
     * Sauvegarde les séances individuelles
     */
    private function saveSeancesFromAI($timetableId, $emploiTemps, $codsec) {
        $stmt = $this->pdo->prepare("
            INSERT INTO seances (timetable_id, codform, codgroupe, type_seance, 
                               codloc, id_enseignant, jour, tranche, codsec, 
                               duree, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
        ");
        
        $seanceCount = 0;
        foreach ($emploiTemps as $creneau => $seances) {
            if (is_array($seances)) {
                foreach ($seances as $seance) {
                    $stmt->execute([
                        $timetableId,
                        $seance['codform'],
                        $seance['codgroupe'],
                        $seance['type_seance'],
                        $seance['codloc'],
                        $seance['id_enseignant'],
                        $seance['jour'],
                        $seance['tranche'],
                        $codsec,
                        $seance['duree'] ?? 1.5
                    ]);
                    $seanceCount++;
                }
            }
        }
        
        return $seanceCount;
    }
    
    /**
     * Analyse des conflits dans un emploi du temps existant
     */
    public function analyzeConflicts($timetableId) {
        try {
            // Récupérer l'emploi du temps
            $timetable = $this->getTimetableById($timetableId);
            if (!$timetable) {
                throw new Exception("Emploi du temps non trouvé");
            }
            
            // Préparer les données pour l'analyse
            $data = $this->prepareTimetableForAnalysis($timetable);
            
            // Appeler le service d'analyse Python
            $analysisResult = $this->callAnalysisService($data);
            
            // Sauvegarder le rapport d'analyse
            $this->saveAnalysisReport($timetableId, $analysisResult);
            
            return [
                'success' => true,
                'timetable_id' => $timetableId,
                'conflicts' => $analysisResult['conflicts'],
                'suggestions' => $analysisResult['suggestions'] ?? [],
                'quality_score' => $analysisResult['quality_score'] ?? 0,
                'detailed_metrics' => $analysisResult['metrics'] ?? []
            ];
            
        } catch (Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage()
            ];
        }
    }
    
    /**
     * Optimise un emploi du temps existant
     */
    public function optimizeExistingTimetable($timetableId, $options = []) {
        try {
            // Récupérer l'emploi du temps existant
            $currentTimetable = $this->getTimetableById($timetableId);
            if (!$currentTimetable) {
                throw new Exception("Emploi du temps non trouvé");
            }
            
            // Préparer les données avec l'état actuel comme point de départ
            $data = $this->prepareAIDataFromExisting($currentTimetable, $options);
            
            // Appeler l'optimisation
            $aiResult = $this->callOptimizationService($data);
            
            // Créer une nouvelle version optimisée
            $newTimetableId = $this->createOptimizedVersion($timetableId, $aiResult);
            
            return [
                'success' => true,
                'original_id' => $timetableId,
                'optimized_id' => $newTimetableId,
                'improvement' => $aiResult['improvement_metrics'] ?? [],
                'comparison' => $this->compareTimetables($timetableId, $newTimetableId)
            ];
            
        } catch (Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage()
            ];
        }
    }
    
    // ========== MÉTHODES D'EXTRACTION DES DONNÉES ==========
    
    private function validateSection($codsec) {
        $stmt = $this->pdo->prepare("SELECT COUNT(*) as count FROM sections WHERE codsec = ?");
        $stmt->execute([$codsec]);
        return $stmt->fetch()['count'] > 0;
    }
    
    private function getSectionData($codsec) {
        $stmt = $this->pdo->prepare("
            SELECT s.*, f.libelle_fil, f.niveau 
            FROM sections s 
            LEFT JOIN filieres f ON s.codfil = f.codfil 
            WHERE s.codsec = ?
        ");
        $stmt->execute([$codsec]);
        return $stmt->fetch(PDO::FETCH_ASSOC);
    }
    
    private function getModulesData($codsec) {
        $stmt = $this->pdo->prepare("
            SELECT f.*, sf.codsec, sf.semestre 
            FROM offre_formation f 
            JOIN section_formations sf ON f.codform = sf.codform 
            WHERE sf.codsec = ?
            ORDER BY f.titre_form
        ");
        $stmt->execute([$codsec]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    
    private function getEnseignantsData() {
        $stmt = $this->pdo->prepare("
            SELECT e.*, 
                   GROUP_CONCAT(CONCAT(em.codform, ':', em.type_seance)) as competences,
                   COALESCE(ep.preference_matin, 0.5) as preference_matin,
                   COALESCE(ep.preference_aprem, 0.5) as preference_aprem,
                   ep.jours_indisponibles,
                   COALESCE(ep.charge_max, 20) as charge_max
            FROM enseignants e
            LEFT JOIN enseignant_module em ON e.id_enseignant = em.id_enseignant
            LEFT JOIN enseignant_preferences ep ON e.id_enseignant = ep.id_enseignant
            WHERE e.actif = 1
            GROUP BY e.id_enseignant
        ");
        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    
    private function getLocauxData() {
        $stmt = $this->pdo->prepare("
            SELECT l.*, 
                   GROUP_CONCAT(le.equipement) as equipements
            FROM local l
            LEFT JOIN local_equipements le ON l.codloc = le.codloc
            WHERE l.actif = 1
            GROUP BY l.codloc
            ORDER BY l.type, l.capacite DESC
        ");
        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    
    private function getGroupesData($codsec) {
        $stmt = $this->pdo->prepare("
            SELECT g.*, 
                   COALESCE(COUNT(DISTINCT e.id_etudiant), 30) as effectif_reel
            FROM groupes g
            LEFT JOIN etudiants e ON g.codgroupe = e.codgroupe
            WHERE g.codsec = ?
            GROUP BY g.codgroupe
            ORDER BY g.groupe
        ");
        $stmt->execute([$codsec]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    
    private function getContraintesData($codsec) {
        $stmt = $this->pdo->prepare("
            SELECT * FROM contraintes_section 
            WHERE codsec = ? OR codsec IS NULL
            ORDER BY priorite DESC
        ");
        $stmt->execute([$codsec]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    
    // ========== MÉTHODES DE VALIDATION ET LOGGING ==========
    
    private function validateAIResult($aiResult) {
        if (!is_array($aiResult)) return false;
        if (!isset($aiResult['success']) || !$aiResult['success']) return false;
        if (!isset($aiResult['emploi_temps']) || !is_array($aiResult['emploi_temps'])) return false;
        if (!isset($aiResult['fitness_final']) || $aiResult['fitness_final'] < 0) return false;
        
        return true;
    }
    
    private function createGenerationLog($codsec, $type) {
        $stmt = $this->pdo->prepare("
            INSERT INTO generation_logs (codsec, type, statut, date_debut, params)
            VALUES (?, ?, 'started', NOW(), ?)
        ");
        $params = json_encode(['section' => $codsec, 'type' => $type]);
        $stmt->execute([$codsec, $type, $params]);
        return $this->pdo->lastInsertId();
    }
    
    private function updateGenerationLog($logId, $statut, $data = []) {
        $stmt = $this->pdo->prepare("
            UPDATE generation_logs 
            SET statut = ?, data_log = ?, date_maj = NOW()
            WHERE id = ?
        ");
        $stmt->execute([$statut, json_encode($data), $logId]);
    }
    
    // ========== MÉTHODES UTILITAIRES ==========
    
    private function deleteExistingTimetable($codsec) {
        // Supprimer les séances existantes
        $stmt = $this->pdo->prepare("DELETE FROM seances WHERE codsec = ?");
        $stmt->execute([$codsec]);
        
        // Marquer l'ancien emploi du temps comme remplacé
        $stmt = $this->pdo->prepare("
            UPDATE emplois_temps 
            SET statut = 'remplace', date_remplacement = NOW() 
            WHERE codsec = ? AND statut IN ('genere_ia', 'actif')
        ");
        $stmt->execute([$codsec]);
    }
    
    private function saveGenerationStats($timetableId, $aiResult) {
        $stmt = $this->pdo->prepare("
            INSERT INTO timetable_stats (timetable_id, fitness_score, nb_iterations, 
                                       temps_execution, nb_conflits, created_at)
            VALUES (?, ?, ?, ?, ?, NOW())
        ");
        
        $stats = $aiResult['solution_summary']['optimization_stats'] ?? [];
        $stmt->execute([
            $timetableId,
            $aiResult['fitness_final'],
            $stats['iterations'] ?? 0,
            $stats['elapsed_time'] ?? 0,
            count($aiResult['solution_summary']['conflicts'] ?? [])
        ]);
    }
    
    private function createQualityReport($timetableId, $aiResult) {
        $metrics = $aiResult['solution_summary']['metrics'] ?? [];
        
        $stmt = $this->pdo->prepare("
            INSERT INTO quality_reports (timetable_id, constraint_satisfaction, 
                                       load_balancing, room_utilization, 
                                       teacher_satisfaction, student_satisfaction, 
                                       global_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NOW())
        ");
        
        $stmt->execute([
            $timetableId,
            $metrics['constraint_satisfaction'] ?? 0,
            $metrics['load_balancing'] ?? 0,
            $metrics['room_utilization'] ?? 0,
            $metrics['teacher_satisfaction'] ?? 0,
            $metrics['student_satisfaction'] ?? 0,
            $aiResult['fitness_final']
        ]);
    }
    
    private function generateBatchSummary($results) {
        $totalFitness = 0;
        $totalConflicts = 0;
        $totalSeances = 0;
        $successCount = 0;
        
        foreach ($results as $result) {
            if ($result['success']) {
                $successCount++;
                $totalFitness += $result['fitness_score'] ?? 0;
                $totalConflicts += count($result['conflicts'] ?? []);
                $totalSeances += $result['nb_seances'] ?? 0;
            }
        }
        
        $avgFitness = $successCount > 0 ? $totalFitness / $successCount : 0;
        
        return [
            'success_rate' => $successCount / count($results),
            'average_fitness' => $avgFitness,
            'total_conflicts' => $totalConflicts,
            'total_seances' => $totalSeances,
            'quality_rating' => $this->getQualityRating($avgFitness)
        ];
    }
    
    private function getQualityRating($fitness) {
        if ($fitness >= 0.9) return 'Excellent';
        if ($fitness >= 0.8) return 'Très Bon';
        if ($fitness >= 0.7) return 'Bon';
        if ($fitness >= 0.6) return 'Acceptable';
        return 'À améliorer';
    }
}

// ==================== API REST ENDPOINTS ====================

/**
 * Contrôleur API pour les endpoints REST
 */
class TimetableAPIController {
    
    private $aiController;
    
    public function __construct() {
        $this->aiController = new TimetableAIController();
    }
    
    public function handleRequest() {
        header('Content-Type: application/json');
        
        $method = $_SERVER['REQUEST_METHOD'];
        $path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
        $pathParts = explode('/', trim($path, '/'));
        
        try {
            switch ($method) {
                case 'POST':
                    return $this->handlePost($pathParts);
                case 'GET':
                    return $this->handleGet($pathParts);
                case 'PUT':
                    return $this->handlePut($pathParts);
                case 'DELETE':
                    return $this->handleDelete($pathParts);
                default:
                    throw new Exception("Méthode non supportée", 405);
            }
        } catch (Exception $e) {
            http_response_code($e->getCode() ?: 500);
            return json_encode([
                'success' => false,
                'error' => $e->getMessage()
            ]);
        }
    }
    
    private function handlePost($pathParts) {
        $input = json_decode(file_get_contents('php://input'), true);
        
        switch ($pathParts[2] ?? '') {
            case 'generate':
                return $this->generateTimetable($input);
            case 'generate-batch':
                return $this->generateBatchTimetables($input);
            case 'analyze':
                return $this->analyzeTimetable($input);
            case 'optimize':
                return $this->optimizeTimetable($input);
            default:
                throw new Exception("Endpoint non trouvé", 404);
        }
    }
    
    private function generateTimetable($input) {
        $codsec = $input['section'] ?? null;
        $options = $input['options'] ?? [];
        
        if (!$codsec) {
            throw new Exception("Section requise", 400);
        }
        
        $result = $this->aiController->generateTimetableAI($codsec, $options);
        return json_encode($result);
    }
    
    private function generateBatchTimetables($input) {
        $sections = $input['sections'] ?? [];
        $options = $input['options'] ?? [];
        
        if (empty($sections)) {
            throw new Exception("Liste de sections requise", 400);
        }
        
        $result = $this->aiController->generateMultipleSections($sections, $options);
        return json_encode($result);
    }
    
    private function analyzeTimetable($input) {
        $timetableId = $input['timetable_id'] ?? null;
        
        if (!$timetableId) {
            throw new Exception("ID emploi du temps requis", 400);
        }
        
        $result = $this->aiController->analyzeConflicts($timetableId);
        return json_encode($result);
    }
    
    private function optimizeTimetable($input) {
        $timetableId = $input['timetable_id'] ?? null;
        $options = $input['options'] ?? [];
        
        if (!$timetableId) {
            throw new Exception("ID emploi du temps requis", 400);
        }
        
        $result = $this->aiController->optimizeExistingTimetable($timetableId, $options);
        return json_encode($result);
    }
}

// ==================== SCRIPT D'UTILISATION ====================

// Configuration des constantes (à définir dans un fichier config.php)
if (!defined('DB_HOST')) {
    define('DB_HOST', 'localhost');
    define('DB_NAME', 'usthb_edt');
    define('DB_USER', 'root');
    define('DB_PASS', '');
}

// Utilisation en mode API
if (isset($_GET['api']) && $_GET['api'] === '1') {
    $api = new TimetableAPIController();
    echo $api->handleRequest();
    exit;
}

// Utilisation directe (exemple)
if (isset($_GET['test']) && $_GET['test'] === '1') {
    try {
        $controller = new TimetableAIController();
        
        // Test de génération pour une section
        $result = $controller->generateTimetableAI('L1-INFO-G1', [
            'prefer_morning' => true,
            'max_daily_hours_student' => 5,
            'optimization_time_limit' => 180
        ]);
        
        echo "<h2>Résultat de génération IA:</h2>";
        echo "<pre>" . json_encode($result, JSON_
