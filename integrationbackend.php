


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
 
</head>
<body>
<?php
/**
 * Système d'intégration Backend PHP avec IA Python
 * Étape 4 : Intégration avec le Backend PHP
 * USTHB - Génération automatique d'emplois du temps
 */

// ==================== CONTRÔLEUR PRINCIPAL IA ====================

class TimetableAIController {
    
    private $pdo;
    private $pythonScriptPath;
    private $modelsPath;
    private $tempPath;
    
    public function __construct() {
        $this->pdo = Database::getConnection();
        $this->pythonScriptPath = __DIR__ . '/../python/timetable_ai_service.py';
        $this->modelsPath = __DIR__ . '/../models/';
        $this->tempPath = __DIR__ . '/../temp/';
        
        // Créer les dossiers s'ils n'existent pas
        if (!is_dir($this->tempPath)) {
            mkdir($this->tempPath, 0755, true);
        }
    }
    
    /**
     * Génère un emploi du temps avec IA pour une section donnée
     */
    public function generateTimetableAI($codsec, $options = []) {
        try {
            // 1. Validation des entrées
            if (!$this->validateSection($codsec)) {
                throw new Exception("Section invalide: $codsec");
            }
            
            // 2. Préparation des données
            $data = $this->prepareAIData($codsec, $options);
            
            // 3. Appel du service IA Python
            $aiResult = $this->callAIService($data);
            
            // 4. Validation du résultat
            if (!$this->validateAIResult($aiResult)) {
                throw new Exception("Résultat IA invalide");
            }
            
            // 5. Sauvegarde en base de données
            $timetableId = $this->saveTimetableFromAI($aiResult, $codsec);
            
            // 6. Retour du résultat
            return [
                'success' => true,
                'timetable_id' => $timetableId,
                'section' => $codsec,
                'fitness_score' => $aiResult['fitness_final'],
                'nb_seances' => $aiResult['nb_seances'],
                'conflicts' => $aiResult['solution_summary']['conflicts'] ?? [],
                'generation_time' => $aiResult['solution_summary']['optimization_stats']['elapsed_time'] ?? 0
            ];
            
        } catch (Exception $e) {
            error_log("Erreur génération IA: " . $e->getMessage());
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'section' => $codsec
            ];
        }
    }
    
    /**
     * Génère des emplois du temps pour plusieurs sections
     */
    public function generateMultipleSections($sections, $options = []) {
        $results = [];
        $overallSuccess = true;
        
        foreach ($sections as $codsec) {
            $result = $this->generateTimetableAI($codsec, $options);
            $results[$codsec] = $result;
            
            if (!$result['success']) {
                $overallSuccess = false;
            }
        }
        
        return [
            'success' => $overallSuccess,
            'results' => $results,
            'total_sections' => count($sections),
            'successful_sections' => count(array_filter($results, fn($r) => $r['success']))
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
        
        // Préférences utilisateur
        $preferences = array_merge([
            'prefer_morning' => true,
            'avoid_thursday' => false,
            'max_daily_hours_teacher' => 6,
            'max_daily_hours_student' => 5,
            'preferred_room_types' => [],
            'optimization_time_limit' => 300 // 5 minutes
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
            ]
        ];
    }
    
    /**
     * Appelle le service IA Python
     */
    private function callAIService($data) {
        // Sauvegarde des données dans un fichier temporaire
        $tempFile = $this->tempPath . 'ai_input_' . uniqid() . '.json';
        file_put_contents($tempFile, json_encode($data, JSON_UNESCAPED_UNICODE));
        
        // Construction de la commande Python
        $command = "python3 {$this->pythonScriptPath} --input {$tempFile} --models {$this->modelsPath}";
        
        // Execution avec timeout
        $descriptors = [
            0 => ["pipe", "r"],
            1 => ["pipe", "w"],
            2 => ["pipe", "w"]
        ];
        
        $process = proc_open($command, $descriptors, $pipes);
        
        if (!is_resource($process)) {
            throw new Exception("Impossible de lancer le processus Python");
        }
        
        // Fermer l'entrée
        fclose($pipes[0]);
        
        // Lire la sortie avec timeout
        stream_set_timeout($pipes[1], 600); // 10 minutes timeout
        $output = stream_get_contents($pipes[1]);
        $error = stream_get_contents($pipes[2]);
        
        fclose($pipes[1]);
        fclose($pipes[2]);
        
        $returnCode = proc_close($process);
        
        // Nettoyage du fichier temporaire
        unlink($tempFile);
        
        if ($returnCode !== 0) {
            throw new Exception("Erreur Python: " . $error);
        }
        
        $result = json_decode($output, true);
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
            // 1. Créer l'enregistrement principal de l'emploi du temps
            $stmt = $this->pdo->prepare("
                INSERT INTO emplois_temps (codsec, statut, date_creation, type_generation, 
                                         fitness_score, nb_seances, metadata)
                VALUES (?, 'genere_ia', NOW(), 'ia_automatique', ?, ?, ?)
            ");
            
            $metadata = json_encode([
                'ai_metrics' => $aiResult['solution_summary']['metrics'] ?? [],
                'conflicts' => $aiResult['solution_summary']['conflicts'] ?? [],
                'optimization_stats' => $aiResult['solution_summary']['optimization_stats'] ?? []
            ]);
            
            $stmt->execute([
                $codsec,
                $aiResult['fitness_final'],
                $aiResult['nb_seances'],
                $metadata
            ]);
            
            $timetableId = $this->pdo->lastInsertId();
            
            // 2. Sauvegarder les séances
            if (isset($aiResult['emploi_temps'])) {
                $this->saveSeancesFromAI($timetableId, $aiResult['emploi_temps'], $codsec);
            }
            
            // 3. Enregistrer les statistiques de génération
            $this->saveGenerationStats($timetableId, $aiResult);
            
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
                               codloc, id_enseignant, jour, tranche, codsec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ");
        
        foreach ($emploiTemps as $creneau => $seances) {
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
                    $codsec
                ]);
            }
        }
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
            
            return [
                'success' => true,
                'timetable_id' => $timetableId,
                'conflicts' => $analysisResult['conflicts'],
                'suggestions' => $analysisResult['suggestions'] ?? [],
                'quality_score' => $analysisResult['quality_score'] ?? 0
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
                'improvement' => $aiResult['improvement_metrics'] ?? []
            ];
            
        } catch (Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage()
            ];
        }
    }
    
    // ========== MÉTHODES D'EXTRACTION DES DONNÉES ==========
    
    private function getSectionData($codsec) {
        $stmt = $this->pdo->prepare("SELECT * FROM sections WHERE codsec = ?");
        $stmt->execute([$codsec]);
        return $stmt->fetch(PDO::FETCH_ASSOC);
    }
    
    private function getModulesData($codsec) {
        $stmt = $this->pdo->prepare("
            SELECT f.*, sf.codsec 
            FROM formations f 
            JOIN section_formations sf ON f.codform = sf.codform 
            WHERE sf.codsec = ?
        ");
        $stmt->execute([$codsec]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    
    private function getEnseignantsData() {
        $stmt = $this->pdo->prepare("
            SELECT e.*, 
                   GROUP_CONCAT(ec.codform) as competences,
                   ep.preference_matin, ep.preference_aprem, ep.jours_indisponibles
            FROM enseignants e



?>

</body>
</html>











