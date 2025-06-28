# Behavioral-Driven Healthcare System for Prostate Cancer Staging
# Implementing DDD and BDD with focus on stakeholder behaviors

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Union, Any
from enum import Enum, auto
import asyncio
from collections import defaultdict
import uuid

# ============================================
# Domain Events (Event Sourcing)
# ============================================

@dataclass
class DomainEvent:
    """Base class for all domain events"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    actor_id: str = ""
    actor_role: str = ""

@dataclass
class PatientAdmitted(DomainEvent):
    patient_id: str = ""
    referring_physician_id: str = ""
    urgency_level: str = ""
    chief_complaint: str = ""

@dataclass
class StudyOrdered(DomainEvent):
    study_id: str = ""
    patient_id: str = ""
    ordered_by: str = ""
    modality: str = ""
    clinical_indication: str = ""
    priority: int = 5

@dataclass
class ImageAcquired(DomainEvent):
    study_id: str = ""
    technologist_id: str = ""
    quality_score: float = 0.0
    notes: str = ""

@dataclass
class StudyAssignedToRadiologist(DomainEvent):
    study_id: str = ""
    radiologist_id: str = ""
    assignment_reason: str = ""  # expertise_match, workload_balance, urgent

@dataclass
class PreliminaryReportGenerated(DomainEvent):
    study_id: str = ""
    ai_confidence: float = 0.0
    findings: Dict = field(default_factory=dict)

@dataclass
class RadiologistReviewStarted(DomainEvent):
    study_id: str = ""
    radiologist_id: str = ""

@dataclass
class FindingAnnotated(DomainEvent):
    study_id: str = ""
    finding_id: str = ""
    annotation_type: str = ""  # agree, disagree, modify
    notes: str = ""

@dataclass
class ReportFinalized(DomainEvent):
    study_id: str = ""
    report_id: str = ""
    turnaround_minutes: int = 0

@dataclass
class CriticalFindingIdentified(DomainEvent):
    study_id: str = ""
    finding: str = ""
    notified_providers: List[str] = field(default_factory=list)

@dataclass
class TreatmentPlanDiscussed(DomainEvent):
    patient_id: str = ""
    participants: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)

# ============================================
# Domain Value Objects
# ============================================

@dataclass(frozen=True)
class Priority:
    """Value object for study priority"""
    STAT = 1  # Emergency
    URGENT = 2  # Within 2 hours
    ROUTINE = 3  # Within 24 hours
    
    value: int
    
    def __post_init__(self):
        if self.value not in [1, 2, 3]:
            raise

# ============================================
# Saga Implementations
# ============================================

class Saga(ABC):
    """Base saga class"""
    @abstractmethod
    async def execute(self):
        pass
    
    @abstractmethod
    async def compensate(self):
        pass

class PatientPreparationSaga(Saga):
    """Saga for patient preparation workflow"""
    def __init__(self, patient: Patient, nurse: Nurse):
        self.patient = patient
        self.nurse = nurse
        self.completed_steps = []
    
    async def execute(self):
        try:
            # Step 1: Verify patient identity
            await self._verify_identity()
            self.completed_steps.append('identity_verified')
            
            # Step 2: Check contraindications
            await self._check_contraindications()
            self.completed_steps.append('contraindications_checked')
            
            # Step 3: Obtain consent
            await self._obtain_consent()
            self.completed_steps.append('consent_obtained')
            
            # Step 4: Prepare for procedure
            await self._prepare_for_procedure()
            self.completed_steps.append('preparation_complete')
            
        except Exception as e:
            raise SagaExecutionError(f"Patient preparation failed: {e}")
    
    async def compensate(self):
        """Rollback completed steps"""
        for step in reversed(self.completed_steps):
            if step == 'preparation_complete':
                await self._cancel_preparation()
            elif step == 'consent_obtained':
                await self._revoke_consent()
            # Other compensation logic

class RadiologyInterpretationSaga(Saga):
    """Saga for radiology interpretation workflow"""
    def __init__(self, study_id: str, radiologist: Radiologist):
        self.study_id = study_id
        self.radiologist = radiologist
        self.interpretation_steps = []
        
    async def execute(self):
        # Step 1: Assign study
        await self._assign_study_to_radiologist()
        
        # Step 2: Review AI findings
        ai_findings = await self._get_ai_findings()
        
        # Step 3: Radiologist interpretation
        interpretation = await self._perform_interpretation(ai_findings)
        
        # Step 4: Quality check
        await self._quality_assurance(interpretation)
        
        # Step 5: Finalize report
        report = await self._finalize_report(interpretation)
        
        # Step 6: Notify referring physician
        await self._notify_referring_physician(report)
    
    async def compensate(self):
        """Rollback interpretation steps"""
        await self._unassign_study()
        await self._clear_preliminary_report()

# ============================================
# Behavioral Interfaces for Different Roles
# ============================================

class RadiologistInterface:
    """Interface specifically designed for radiologist workflow"""
    def __init__(self, radiologist: Radiologist):
        self.radiologist = radiologist
        self.current_session = None
        self.voice_commands_enabled = True
        self.eye_tracking_enabled = False
        
    def start_reading_session(self):
        """Initialize reading session with personalized setup"""
        self.current_session = ReadingSession(
            radiologist=self.radiologist,
            start_time=datetime.now()
        )
        
        # Load personalized settings
        self._load_hanging_protocols()
        self._configure_voice_commands()
        self._setup_ambient_lighting()
        
        return self.current_session
    
    def present_next_study(self) -> Optional[Study]:
        """Present next study based on radiologist preferences"""
        # Get prioritized worklist
        worklist = self._get_prioritized_worklist()
        
        if not worklist:
            return None
        
        next_study = worklist[0]
        
        # Pre-load relevant priors
        self._preload_comparison_studies(next_study)
        
        # Display with preferred layout
        self._apply_hanging_protocol(next_study)
        
        # Show AI findings if preferred
        if self.radiologist.workelist_preferences.notification_preferences.get('show_ai_findings', True):
            self._display_ai_overlay(next_study)
        
        return next_study
    
    def capture_finding(self, finding_type: str, location: Dict, 
                       severity: str, voice_note: Optional[str] = None):
        """Capture finding with multimodal input"""
        finding = Finding(
            type=finding_type,
            location=location,
            severity=severity,
            annotated_by=self.radiologist.id,
            timestamp=datetime.now()
        )
        
        if voice_note:
            finding.voice_transcription = self._transcribe_voice(voice_note)
        
        # Auto-generate measurement if applicable
        if finding_type in ['nodule', 'mass', 'lesion']:
            finding.measurements = self._auto_measure(location)
        
        self.current_session.add_finding(finding)
        
        # Check if finding is critical
        if self._is_critical_finding(finding):
            self._trigger_critical_finding_workflow(finding)
    
    def _get_prioritized_worklist(self) -> List[Study]:
        """Get studies prioritized by multiple factors"""
        all_studies = self._fetch_assigned_studies()
        
        # Score each study
        scored_studies = []
        for study in all_studies:
            score = 0
            
            # Priority score
            score += (4 - study.priority.value) * 10
            
            # Subspecialty match
            if study.modality in self.radiologist.subspecialties[:2]:
                score += 5
            
            # Time waiting
            wait_time = (datetime.now() - study.created_at).total_seconds() / 3600
            score += min(wait_time * 2, 10)
            
            # Referring physician preference
            if self._is_preferred_referrer(study):
                score += 3
                
            scored_studies.append((score, study))
        
        # Sort by score
        scored_studies.sort(key=lambda x: x[0], reverse=True)
        
        return [study for _, study in scored_studies]

class NurseInterface:
    """Interface designed for nursing workflow"""
    def __init__(self, nurse: Nurse):
        self.nurse = nurse
        self.mobile_device = MobileDevice()
        self.barcode_scanner = BarcodeScanner()
        
    def patient_check_in(self, patient_barcode: str):
        """Mobile-first patient check-in"""
        # Scan patient wristband
        patient_id = self.barcode_scanner.scan(patient_barcode)
        
        # Pull up patient info
        patient = self._get_patient(patient_id)
        
        # Show relevant alerts
        alerts = []
        if patient.allergies:
            alerts.append(Alert("ALLERGIES", patient.allergies, "red"))
        
        if patient.fall_risk:
            alerts.append(Alert("FALL RISK", "High", "yellow"))
        
        # Get today's procedures
        procedures = self._get_scheduled_procedures(patient_id)
        
        return PatientCheckInView(
            patient=patient,
            alerts=alerts,
            procedures=procedures
        )
    
    def medication_administration(self, patient_id: str, medication_barcode: str):
        """Five rights verification for medication"""
        # Scan medication
        medication = self.barcode_scanner.scan_medication(medication_barcode)
        
        # Perform five rights check
        verification = FiveRightsVerification(
            right_patient=self._verify_patient(patient_id),
            right_medication=self._verify_medication(medication),
            right_dose=self._verify_dose(medication),
            right_route=self._verify_route(medication),
            right_time=self._verify_time(medication)
        )
        
        if not verification.all_verified():
            return MedicationError(verification.get_errors())
        
        # Document administration
        self._document_medication_admin(patient_id, medication, self.nurse.id)
        
        return MedicationSuccess()

class UrologistInterface:
    """Interface designed for urologist workflow"""
    def __init__(self, urologist: Urologist):
        self.urologist = urologist
        self.decision_support = ClinicalDecisionSupport()
        
    def review_patient_dashboard(self, patient_id: str):
        """Comprehensive patient dashboard"""
        patient = self._get_patient(patient_id)
        
        # Get all relevant data
        dashboard = PatientDashboard(
            demographics=patient.get_demographics(),
            psa_trend=self._get_psa_trend(patient_id),
            imaging_timeline=self._get_imaging_timeline(patient_id),
            pathology_results=self._get_pathology_results(patient_id),
            current_medications=self._get_medications(patient_id),
            treatment_history=self._get_treatment_history(patient_id)
        )
        
        # Add decision support
        dashboard.recommendations = self.decision_support.provide_recommendations(
            dashboard.get_latest_staging(),
            dashboard.get_patient_factors()
        )
        
        # Risk calculators
        dashboard.risk_scores = {
            'nomogram': self._calculate_nomogram(dashboard),
            'decipher': self._get_genomic_score(patient_id),
            'capra': self._calculate_capra_score(dashboard)
        }
        
        return dashboard
    
    def plan_treatment(self, patient_id: str, staging: Dict):
        """Interactive treatment planning"""
        planner = TreatmentPlanner(patient_id, staging)
        
        # Get treatment options
        options = planner.get_treatment_options()
        
        # For each option, show:
        # - Expected outcomes
        # - Side effects profile  
        # - Quality of life impact
        # - Cost considerations
        
        for option in options:
            option.outcomes = self._simulate_outcomes(option)
            option.qol_impact = self._assess_qol_impact(option)
            option.cost_analysis = self._calculate_costs(option)
        
        return TreatmentPlanView(options)

# ============================================
# Collaborative Features
# ============================================

class TumorBoard:
    """Multi-disciplinary tumor board"""
    def __init__(self):
        self.participants = []
        self.cases = []
        self.decisions = []
        
    def schedule_case(self, patient: Patient, presenter: HealthcareProvider):
        """Schedule case for tumor board review"""
        case = TumorBoardCase(
            patient=patient,
            presenter=presenter,
            scheduled_date=self._next_meeting_date(),
            materials_needed=[
                'Imaging studies',
                'Pathology slides', 
                'Clinical summary',
                'Treatment options'
            ]
        )
        
        self.cases.append(case)
        
        # Notify participants
        for participant in self.participants:
            self._notify_participant(participant, case)
        
        return case
    
    def conduct_review(self, case: 'TumorBoardCase'):
        """Conduct tumor board review"""
        review_session = TumorBoardSession(
            case=case,
            participants=self.participants,
            start_time=datetime.now()
        )
        
        # Each specialist provides input
        radiologist_input = self._get_radiologist_input(case)
        pathologist_input = self._get_pathologist_input(case)
        urologist_input = self._get_urologist_input(case)
        oncologist_input = self._get_oncologist_input(case)
        
        # Consensus building
        consensus = self._build_consensus([
            radiologist_input,
            pathologist_input,
            urologist_input,
            oncologist_input
        ])
        
        # Document decision
        decision = TumorBoardDecision(
            case_id=case.id,
            consensus_recommendation=consensus,
            dissenting_opinions=self._get_dissenting_opinions(),
            follow_up_plan=self._create_follow_up_plan()
        )
        
        self.decisions.append(decision)
        
        return decision

class CommunicationHub:
    """Secure communication between healthcare providers"""
    def __init__(self):
        self.channels = {}
        self.message_store = MessageStore()
        
    def create_case_channel(self, patient_id: str, providers: List[HealthcareProvider]):
        """Create secure communication channel for patient case"""
        channel = SecureChannel(
            id=f"case_{patient_id}",
            participants=providers,
            encryption_enabled=True,
            audit_enabled=True
        )
        
        self.channels[channel.id] = channel
        
        # Send initial message
        channel.post_message(
            author="System",
            content=f"Case discussion channel created for patient {patient_id}",
            priority="info"
        )
        
        return channel
    
    def send_urgent_consultation(self, from_provider: HealthcareProvider,
                               to_provider: HealthcareProvider,
                               patient_id: str,
                               message: str):
        """Send urgent consultation request"""
        consultation = UrgentConsultation(
            from_id=from_provider.id,
            to_id=to_provider.id,
            patient_id=patient_id,
            message=message,
            created_at=datetime.now(),
            requires_response_by=datetime.now() + timedelta(hours=2)
        )
        
        # Multiple notification methods
        self._send_page(to_provider, consultation)
        self._send_push_notification(to_provider, consultation)
        self._send_sms(to_provider, consultation)
        
        # Track response time
        self._start_response_timer(consultation)
        
        return consultation

# ============================================
# Quality and Safety Systems
# ============================================

class QualityAssuranceSystem:
    """Monitors and ensures quality across the system"""
    def __init__(self):
        self.metrics = QualityMetrics()
        self.peer_review = PeerReviewSystem()
        self.safety_monitor = PatientSafetyMonitor()
        
    def monitor_radiologist_performance(self, radiologist: Radiologist):
        """Track radiologist performance metrics"""
        metrics = {
            'turnaround_time': self._calculate_average_tat(radiologist),
            'amendment_rate': self._calculate_amendment_rate(radiologist),
            'peer_review_score': self.peer_review.get_score(radiologist),
            'critical_finding_communication': self._track_critical_comms(radiologist),
            'productivity': self._calculate_rvu_productivity(radiologist)
        }
        
        # Flag if below thresholds
        alerts = []
        if metrics['turnaround_time'] > 120:  # minutes
            alerts.append("TAT exceeds target")
        
        if metrics['amendment_rate'] > 0.02:  # 2%
            alerts.append("High amendment rate")
            
        return PerformanceReport(radiologist, metrics, alerts)
    
    def detect_safety_events(self, patient: Patient):
        """Detect potential safety events"""
        safety_checks = [
            self._check_contrast_allergy(patient),
            self._check_kidney_function(patient),
            self._check_pregnancy_status(patient),
            self._check_implant_compatibility(patient),
            self._check_claustrophobia_protocol(patient)
        ]
        
        risks = [check for check in safety_checks if check.has_risk()]
        
        if risks:
            event = PatientSafetyEvent(
                patient_id=patient.id,
                risks=risks,
                severity=max(r.severity for r in risks)
            )
            
            self.safety_monitor.log_event(event)
            self._notify_safety_team(event)
            
        return risks

# ============================================
# Learning and Improvement System
# ============================================

class ContinuousLearningSystem:
    """System for continuous improvement and learning"""
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.outcome_tracker = OutcomeTracker()
        self.education_platform = EducationPlatform()
        
    def collect_radiologist_feedback(self, radiologist: Radiologist, 
                                   study: Study, ai_findings: Dict):
        """Collect feedback on AI performance"""
        feedback = RadiologistFeedback(
            radiologist_id=radiologist.id,
            study_id=study.id,
            ai_accuracy=self._rate_ai_accuracy(ai_findings),
            missed_findings=self._identify_missed_findings(ai_findings),
            false_positives=self._identify_false_positives(ai_findings),
            overall_utility=self._rate_utility()
        )
        
        self.feedback_collector.store(feedback)
        
        # Use feedback to improve model
        if feedback.has_significant_discrepancy():
            self._queue_for_model_retraining(study, feedback)
    
    def track_patient_outcomes(self, patient: Patient, treatment: 'Treatment'):
        """Track long-term patient outcomes"""
        outcome = PatientOutcome(
            patient_id=patient.id,
            treatment_type=treatment.type,
            follow_up_period=self._calculate_follow_up_period(treatment),
            metrics={
                'psa_response': self._track_psa_response(patient),
                'imaging_response': self._track_imaging_response(patient),
                'quality_of_life': self._assess_qol(patient),
                'survival_data': self._track_survival(patient)
            }
        )
        
        self.outcome_tracker.record(outcome)
        
        # Contribute to research
        if patient.consented_for_research:
            self._submit_to_registry(outcome)

# ============================================
# Real-world Testing Scenarios
# ============================================

class ClinicalScenarioTester:
    """Test system with real-world clinical scenarios"""
    
    @staticmethod
    def test_emergency_workflow():
        """Test STAT study workflow"""
        scenario = EmergencyScenario(
            description="Patient with acute urinary retention and suspected malignancy"
        )
        
        # Create actors
        ed_physician = EmergencyPhysician("ED001", "Dr. Emergency")
        radiologist = Radiologist("R001", "Dr. Rad", ["MD"], ["CT", "MRI"])
        radiologist.on_call = True
        
        # Patient arrives in ED
        patient = Patient("P_EMERGENCY", "MRN999")
        patient.presentation = "Acute urinary retention, elevated PSA 45"
        
        # ED orders STAT study
        study_order = ed_physician.order_stat_study(
            patient=patient,
            modality="CT Abdomen/Pelvis",
            indication="Rule out obstructing mass"
        )
        
        # Verify STAT handling
        assert study_order.priority.value == Priority.STAT
        assert radiologist.receives_stat_notification(study_order)
        assert radiologist.interrupts_current_study()
        
        # Radiologist reads immediately
        start_time = datetime.now()
        report = radiologist.read_stat_study(study_order)
        turnaround = (datetime.now() - start_time).seconds / 60
        
        assert turnaround < 30  # Must be read within 30 minutes
        assert ed_physician.receives_critical_results(report)
        
        return scenario.passed()
    
    @staticmethod  
    def test_multidisciplinary_collaboration():
        """Test tumor board workflow"""
        # Create multidisciplinary team
        urologist = Urologist("U001", "Dr. Uro", ["MD", "FACS"])
        radiologist = Radiologist("R001", "Dr. Rad", ["MD"], ["MRI"]) 
        pathologist = Pathologist("P001", "Dr. Path", ["MD"])
        oncologist = Oncologist("O001", "Dr. Onc", ["MD"])
        
        # Complex case
        patient = Patient("P_COMPLEX", "MRN_TB001")
        patient.staging = "T3b N1 M0"
        patient.gleason = "4+5=9"
        patient.psa = 28.5
        
        # Schedule tumor board
        tumor_board = TumorBoard()
        tumor_board.add_participants([urologist, radiologist, pathologist, oncologist])
        
        case = tumor_board.schedule_case(patient, urologist)
        
        # Each specialist reviews
        rad_review = radiologist.review_for_tumor_board(patient)
        path_review = pathologist.review_slides(patient.biopsy_slides)
        
        # Conduct tumor board
        decision = tumor_board.conduct_review(case)
        
        # Verify collaboration
        assert len(decision.participant_inputs) == 4
        assert decision.has_consensus()
        assert patient.treatment_plan.is_multidisciplinary()
        
        return True

# ============================================
# Performance Monitoring
# ============================================

class SystemPerformanceMonitor:
    """Monitor system performance from user perspective"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def track_user_satisfaction(self, user: HealthcareProvider, 
                              interaction: 'UserInteraction'):
        """Track user satisfaction metrics"""
        satisfaction = UserSatisfactionMetric(
            user_id=user.id,
            user_role=type(user).__name__,
            task_completed=interaction.task_name,
            time_to_complete=interaction.duration,
            clicks_required=interaction.click_count,
            errors_encountered=interaction.error_count,
            satisfaction_score=interaction.get_satisfaction_score()
        )
        
        self.metrics[user.id].append(satisfaction)
        
        # Alert if satisfaction drops
        if satisfaction.satisfaction_score < 3:
            self._alert_ux_team(satisfaction)
    
    def generate_usability_report(self) -> Dict:
        """Generate usability report by role"""
        report = {}
        
        for role in ['Radiologist', 'Nurse', 'Urologist', 'Technologist']:
            role_metrics = [m for m in self.metrics.values() 
                          if m.user_role == role]
            
            report[role] = {
                'average_task_time': self._avg_task_time(role_metrics),
                'common_errors': self._common_errors(role_metrics),
                'satisfaction_trend': self._satisfaction_trend(role_metrics),
                'feature_usage': self._feature_usage(role_metrics),
                'pain_points': self._identify_pain_points(role_metrics)
            }
        
        return report

if __name__ == "__main__":
    # Run behavioral tests
    print("Running Behavioral Tests...")
    
    # Test radiologist behaviors
    assert RadiologistBehaviors.reading_urgent_study().execute()
    assert RadiologistBehaviors.fatigue_management().execute()
    
    # Test nurse behaviors  
    assert NurseBehaviors.patient_preparation().execute()
    
    # Test clinical scenarios
    assert ClinicalScenarioTester.test_emergency_workflow()
    assert ClinicalScenarioTester.test_multidisciplinary_collaboration()
    
    print("All behavioral tests passed!")
    
    # Demonstrate workflow
    print("\nDemonstrating Patient Journey...")
    
    workflow = ClinicalWorkflow()
    patient = Patient("P001", "MRN12345")
    
    # Run async workflow
    import asyncio
    asyncio.run(workflow.patient_journey(patient, "Elevated PSA 12.5")) ValueError("Invalid priority")
    
    def expected_turnaround(self) -> timedelta:
        turnarounds = {
            1: timedelta(minutes=30),
            2: timedelta(hours=2),
            3: timedelta(hours=24)
        }
        return turnarounds[self.value]

@dataclass(frozen=True)
class ClinicalContext:
    """Value object for clinical context"""
    symptoms: List[str]
    prior_studies: List[str]
    relevant_history: str
    medications: List[str]
    allergies: List[str]
    
    def requires_contrast(self) -> bool:
        """Determine if contrast is needed based on context"""
        contrast_indications = ['suspected metastasis', 'post-treatment', 'rising PSA']
        return any(indication in self.relevant_history.lower() 
                  for indication in contrast_indications)

@dataclass(frozen=True)
class WorklistPreferences:
    """Radiologist workelist preferences"""
    subspecialties: List[str]
    complexity_preference: str  # simple, moderate, complex
    max_daily_studies: int
    preferred_hours: tuple  # (start_hour, end_hour)
    notification_preferences: Dict[str, bool]

# ============================================
# Domain Entities
# ============================================

class HealthcareProvider:
    """Base class for all healthcare providers"""
    def __init__(self, provider_id: str, name: str, credentials: List[str]):
        self.id = provider_id
        self.name = name
        self.credentials = credentials
        self.current_workload = 0
        self.shift_start = None
        self.shift_end = None
        self.break_times = []
        self._event_handlers = defaultdict(list)
    
    def subscribe_to_event(self, event_type: type, handler):
        """Subscribe to domain events"""
        self._event_handlers[event_type].append(handler)
    
    def handle_event(self, event: DomainEvent):
        """Handle domain events"""
        for handler in self._event_handlers[type(event)]:
            handler(event)

class Radiologist(HealthcareProvider):
    """Radiologist entity with specific behaviors"""
    def __init__(self, provider_id: str, name: str, credentials: List[str], 
                 subspecialties: List[str]):
        super().__init__(provider_id, name, credentials)
        self.subspecialties = subspecialties
        self.workelist_preferences = None
        self.reading_speed_wpm = 150  # words per minute
        self.accuracy_rate = 0.95
        self.fatigue_factor = 1.0
        self.studies_read_today = 0
        self.active_study = None
        self.peer_review_queue = []
    
    def can_read_study(self, study: 'Study') -> bool:
        """Check if radiologist can read this study"""
        # Check credentials
        if study.modality not in self.subspecialties:
            return False
        
        # Check workload
        if self.current_workload >= self.workelist_preferences.max_daily_studies:
            return False
        
        # Check shift timing
        current_hour = datetime.now().hour
        if not (self.workelist_preferences.preferred_hours[0] <= 
                current_hour <= self.workelist_preferences.preferred_hours[1]):
            return False
        
        # Check fatigue
        if self.fatigue_factor < 0.7:
            return False
        
        return True
    
    def estimate_reading_time(self, study: 'Study') -> timedelta:
        """Estimate time to read a study based on complexity"""
        base_times = {
            'simple': 10,
            'moderate': 20,
            'complex': 30
        }
        
        # Adjust for fatigue
        adjusted_time = base_times[study.complexity] * (2 - self.fatigue_factor)
        
        # Adjust for experience with modality
        if study.modality in self.subspecialties[:2]:  # Primary subspecialties
            adjusted_time *= 0.8
        
        return timedelta(minutes=int(adjusted_time))
    
    def update_fatigue(self):
        """Update fatigue based on workload"""
        # Fatigue increases with studies read
        self.fatigue_factor = max(0.5, 1.0 - (self.studies_read_today * 0.02))
        
        # Recovery during breaks
        if self._is_break_time():
            self.fatigue_factor = min(1.0, self.fatigue_factor + 0.1)
    
    def _is_break_time(self) -> bool:
        """Check if current time is break time"""
        current_time = datetime.now().time()
        for break_start, break_end in self.break_times:
            if break_start <= current_time <= break_end:
                return True
        return False

class Urologist(HealthcareProvider):
    """Urologist entity with specific behaviors"""
    def __init__(self, provider_id: str, name: str, credentials: List[str]):
        super().__init__(provider_id, name, credentials)
        self.patient_panel = []
        self.surgery_schedule = []
        self.clinic_schedule = []
        self.on_call = False
        self.response_time_preference = timedelta(hours=4)  # Preferred response time
    
    def review_staging_report(self, report: 'StagingReport') -> 'TreatmentDecision':
        """Review staging report and make treatment decision"""
        decision = TreatmentDecision(
            provider_id=self.id,
            patient_id=report.patient_id,
            timestamp=datetime.utcnow()
        )
        
        # Decision logic based on staging
        if report.risk_group == "Low":
            decision.recommendation = "Active Surveillance"
            decision.follow_up_interval = timedelta(days=180)
        elif report.risk_group == "Intermediate":
            decision.recommendation = "Consider definitive treatment"
            decision.options = ["Radical prostatectomy", "Radiation therapy"]
            decision.follow_up_interval = timedelta(days=30)
        else:  # High risk
            decision.recommendation = "Urgent treatment needed"
            decision.options = ["Multimodal therapy"]
            decision.follow_up_interval = timedelta(days=7)
            decision.urgent_consultation = True
        
        return decision
    
    def request_additional_studies(self, patient_id: str, 
                                 current_findings: Dict) -> List['StudyOrder']:
        """Determine if additional studies are needed"""
        additional_studies = []
        
        # Logic for additional studies
        if current_findings.get('psa_velocity', 0) > 0.75:
            additional_studies.append(
                StudyOrder(
                    modality="Bone Scan",
                    indication="High PSA velocity",
                    priority=Priority(Priority.URGENT)
                )
            )
        
        if current_findings.get('stage', '') >= 'T3':
            additional_studies.append(
                StudyOrder(
                    modality="CT Abdomen/Pelvis",
                    indication="Advanced local disease",
                    priority=Priority(Priority.URGENT)
                )
            )
        
        return additional_studies

class Nurse(HealthcareProvider):
    """Nurse entity with specific behaviors"""
    def __init__(self, provider_id: str, name: str, credentials: List[str], 
                 unit: str):
        super().__init__(provider_id, name, credentials)
        self.unit = unit
        self.patient_assignments = []
        self.task_queue = []
        self.medication_admin_certified = True
    
    def prepare_patient_for_study(self, patient: 'Patient', 
                                 study: 'Study') -> 'PreparationChecklist':
        """Prepare patient for imaging study"""
        checklist = PreparationChecklist(
            patient_id=patient.id,
            study_id=study.id,
            nurse_id=self.id
        )
        
        # Standard preparation steps
        checklist.verify_patient_identity()
        checklist.check_allergies()
        checklist.verify_consent()
        
        # Study-specific preparation
        if study.requires_contrast():
            checklist.check_renal_function()
            checklist.start_iv_access()
            checklist.verify_contrast_consent()
        
        if study.modality == "MRI":
            checklist.screen_for_metal()
            checklist.provide_gown()
            checklist.remove_jewelry()
        
        return checklist
    
    def monitor_patient_post_procedure(self, patient: 'Patient', 
                                     duration: timedelta) -> List['VitalSigns']:
        """Monitor patient after procedure"""
        vital_checks = []
        check_intervals = [0, 15, 30, 60]  # minutes
        
        for interval in check_intervals:
            if interval <= duration.total_seconds() / 60:
                vitals = VitalSigns(
                    patient_id=patient.id,
                    timestamp=datetime.now() + timedelta(minutes=interval),
                    blood_pressure=(120, 80),
                    heart_rate=72,
                    temperature=98.6,
                    oxygen_saturation=98
                )
                vital_checks.append(vitals)
        
        return vital_checks

class Technologist(HealthcareProvider):
    """Imaging technologist entity"""
    def __init__(self, provider_id: str, name: str, credentials: List[str], 
                 modalities: List[str]):
        super().__init__(provider_id, name, credentials)
        self.certified_modalities = modalities
        self.equipment_proficiency = {}
        self.quality_score = 0.95
        self.scan_times = defaultdict(list)  # Track scan duration by type
    
    def perform_quality_check(self, images: List['Image']) -> 'QualityAssessment':
        """Perform quality check on acquired images"""
        assessment = QualityAssessment(
            technologist_id=self.id,
            timestamp=datetime.utcnow()
        )
        
        for image in images:
            # Check technical parameters
            if image.check_motion_artifact():
                assessment.add_issue("Motion artifact detected", image.id)
            
            if not image.check_coverage():
                assessment.add_issue("Incomplete anatomical coverage", image.id)
            
            if not image.check_contrast():
                assessment.add_issue("Suboptimal contrast", image.id)
            
            # Position verification
            if not self.verify_positioning(image):
                assessment.add_issue("Positioning error", image.id)
        
        assessment.calculate_overall_score()
        return assessment
    
    def optimize_protocol(self, patient: 'Patient', 
                         study_type: str) -> 'ImagingProtocol':
        """Optimize imaging protocol based on patient factors"""
        protocol = ImagingProtocol(study_type)
        
        # Adjust for patient factors
        if patient.bmi > 30:
            protocol.increase_penetration()
        
        if patient.has_implants:
            protocol.add_metal_suppression()
        
        if patient.claustrophobic and study_type == "MRI":
            protocol.use_fast_sequences()
        
        return protocol

# ============================================
# Domain Aggregates
# ============================================

class Study:
    """Study aggregate root"""
    def __init__(self, study_id: str, patient_id: str, modality: str):
        self.id = study_id
        self.patient_id = patient_id
        self.modality = modality
        self.status = "Ordered"
        self.priority = Priority(Priority.ROUTINE)
        self.images = []
        self.reports = []
        self.assigned_radiologist = None
        self.clinical_context = None
        self.events = []
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.complexity = "moderate"
    
    def assign_to_radiologist(self, radiologist: Radiologist, 
                            assignment_service: 'AssignmentService'):
        """Assign study to radiologist using domain service"""
        if assignment_service.can_assign(self, radiologist):
            self.assigned_radiologist = radiologist.id
            self.status = "Assigned"
            
            event = StudyAssignedToRadiologist(
                study_id=self.id,
                radiologist_id=radiologist.id,
                assignment_reason=assignment_service.get_assignment_reason()
            )
            self.events.append(event)
            radiologist.handle_event(event)
    
    def add_preliminary_findings(self, ai_service: 'AIAnalysisService'):
        """Add AI preliminary findings"""
        findings = ai_service.analyze_study(self)
        
        event = PreliminaryReportGenerated(
            study_id=self.id,
            ai_confidence=findings.confidence,
            findings=findings.to_dict()
        )
        self.events.append(event)
        
        # Check for critical findings
        if findings.has_critical_findings():
            self.escalate_critical_findings(findings.critical_findings)
    
    def escalate_critical_findings(self, findings: List[str]):
        """Escalate critical findings"""
        self.priority = Priority(Priority.STAT)
        self.status = "Critical"
        
        event = CriticalFindingIdentified(
            study_id=self.id,
            finding="; ".join(findings)
        )
        self.events.append(event)
    
    def complete_study(self, report: 'Report'):
        """Complete the study with final report"""
        self.reports.append(report)
        self.status = "Completed"
        self.completed_at = datetime.utcnow()
        
        turnaround = int((self.completed_at - self.created_at).total_seconds() / 60)
        
        event = ReportFinalized(
            study_id=self.id,
            report_id=report.id,
            turnaround_minutes=turnaround
        )
        self.events.append(event)
    
    def requires_contrast(self) -> bool:
        """Determine if study requires contrast"""
        return self.clinical_context and self.clinical_context.requires_contrast()

class Patient:
    """Patient aggregate root"""
    def __init__(self, patient_id: str, mrn: str):
        self.id = patient_id
        self.mrn = mrn  # Medical Record Number
        self.studies = []
        self.care_team = []
        self.treatment_plans = []
        self.risk_factors = []
        self.allergies = []
        self.bmi = 0
        self.has_implants = False
        self.claustrophobic = False
        
    def add_to_care_team(self, provider: HealthcareProvider, role: str):
        """Add provider to care team"""
        self.care_team.append({
            'provider': provider,
            'role': role,
            'added_date': datetime.utcnow()
        })
    
    def get_active_treatment_plan(self) -> Optional['TreatmentPlan']:
        """Get current active treatment plan"""
        active_plans = [p for p in self.treatment_plans if p.is_active()]
        return active_plans[0] if active_plans else None

# ============================================
# Domain Services
# ============================================

class AssignmentService:
    """Domain service for study assignment"""
    def __init__(self):
        self.assignment_rules = []
        self.workload_balancer = WorkloadBalancer()
        self._last_assignment_reason = ""
    
    def can_assign(self, study: Study, radiologist: Radiologist) -> bool:
        """Check if study can be assigned to radiologist"""
        # Check basic eligibility
        if not radiologist.can_read_study(study):
            return False
        
        # Check workload balance
        if not self.workload_balancer.is_balanced_assignment(radiologist):
            return False
        
        # Subspecialty matching for complex cases
        if study.complexity == "complex":
            if study.modality not in radiologist.subspecialties[:1]:  # Primary subspecialty
                return False
        
        self._last_assignment_reason = "expertise_match"
        return True
    
    def find_best_radiologist(self, study: Study, 
                            available_radiologists: List[Radiologist]) -> Optional[Radiologist]:
        """Find best radiologist for study"""
        candidates = [r for r in available_radiologists if self.can_assign(study, r)]
        
        if not candidates:
            return None
        
        # Score candidates
        scores = {}
        for radiologist in candidates:
            score = 0
            
            # Subspecialty match
            if study.modality in radiologist.subspecialties:
                score += 10
            
            # Workload (prefer less loaded)
            score += (10 - radiologist.current_workload)
            
            # Fatigue factor
            score += radiologist.fatigue_factor * 5
            
            # Previous experience with patient
            if self._has_read_for_patient(radiologist, study.patient_id):
                score += 3
            
            scores[radiologist] = score
        
        # Return radiologist with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_assignment_reason(self) -> str:
        return self._last_assignment_reason
    
    def _has_read_for_patient(self, radiologist: Radiologist, patient_id: str) -> bool:
        """Check if radiologist has previously read for this patient"""
        # Implementation would check historical data
        return False

class WorkloadBalancer:
    """Service to balance workload across providers"""
    def __init__(self):
        self.target_daily_studies = {
            'simple': 30,
            'moderate': 20,
            'complex': 10
        }
    
    def is_balanced_assignment(self, radiologist: Radiologist) -> bool:
        """Check if assignment maintains workload balance"""
        # Check if radiologist is significantly below average workload
        return radiologist.current_workload < radiologist.workelist_preferences.max_daily_studies

class NotificationService:
    """Service for handling notifications"""
    def __init__(self):
        self.channels = {
            'pager': PagerChannel(),
            'email': EmailChannel(),
            'app': AppNotificationChannel(),
            'sms': SMSChannel()
        }
    
    def notify_critical_finding(self, finding: CriticalFindingIdentified, 
                              recipients: List[HealthcareProvider]):
        """Send critical finding notifications"""
        for recipient in recipients:
            # Determine notification preference
            if isinstance(recipient, Urologist) and recipient.on_call:
                self.channels['pager'].send(recipient, finding)
            elif isinstance(recipient, Radiologist):
                self.channels['app'].send(recipient, finding)
            
            # Always send email for documentation
            self.channels['email'].send(recipient, finding)
    
    def notify_study_ready(self, study: Study, radiologist: Radiologist):
        """Notify radiologist that study is ready"""
        if radiologist.workelist_preferences.notification_preferences.get('study_ready', True):
            self.channels['app'].send(
                radiologist, 
                f"New {study.priority.name} study ready for review"
            )

class ClinicalDecisionSupport:
    """Service providing clinical decision support"""
    def __init__(self):
        self.guidelines = self._load_clinical_guidelines()
        self.risk_calculators = {
            'nomogram': ProstateNomogram(),
            'risk_stratification': RiskStratificationTool()
        }
    
    def provide_recommendations(self, staging_results: Dict, 
                              patient_factors: Dict) -> List[str]:
        """Provide evidence-based recommendations"""
        recommendations = []
        
        # Calculate risk scores
        risk_score = self.risk_calculators['nomogram'].calculate(
            staging_results, patient_factors
        )
        
        # Apply guidelines
        if risk_score.category == "Low":
            recommendations.extend([
                "Consider active surveillance per NCCN guidelines",
                "PSA monitoring every 6 months",
                "Repeat biopsy in 12-18 months"
            ])
        elif risk_score.category == "Intermediate":
            recommendations.extend([
                "Discuss definitive treatment options",
                "Consider genomic testing for risk refinement",
                "Baseline bone scan if Gleason 4+3"
            ])
        else:
            recommendations.extend([
                "Multimodal therapy recommended",
                "Staging scans indicated",
                "Refer to multidisciplinary team"
            ])
        
        return recommendations
    
    def _load_clinical_guidelines(self) -> Dict:
        """Load current clinical guidelines"""
        return {
            'NCCN': 'version_2024',
            'AUA': 'version_2023',
            'EAU': 'version_2024'
        }

# ============================================
# Behavioral Specifications (BDD)
# ============================================

class BehaviorSpecification:
    """Base class for behavior specifications"""
    def __init__(self, title: str):
        self.title = title
        self.given_context = {}
        self.when_action = None
        self.then_outcomes = []
    
    def given(self, **context):
        """Set up the context"""
        self.given_context.update(context)
        return self
    
    def when(self, action):
        """Define the action"""
        self.when_action = action
        return self
    
    def then(self, outcome):
        """Define expected outcome"""
        self.then_outcomes.append(outcome)
        return self
    
    def execute(self) -> bool:
        """Execute the specification"""
        # Set up context
        for key, value in self.given_context.items():
            globals()[key] = value
        
        # Execute action
        if self.when_action:
            result = self.when_action()
        
        # Verify outcomes
        for outcome in self.then_outcomes:
            if not outcome():
                return False
        
        return True

# Radiologist Behavior Specifications
class RadiologistBehaviors:
    @staticmethod
    def reading_urgent_study():
        return (BehaviorSpecification("Radiologist reads urgent study")
                .given(
                    radiologist=Radiologist("R001", "Dr. Smith", ["MD", "Board Certified"], ["MRI", "CT"]),
                    study=Study("S001", "P001", "MRI"),
                    study_priority=Priority(Priority.URGENT)
                )
                .when(lambda: radiologist.handle_event(
                    StudyAssignedToRadiologist(study_id=study.id, radiologist_id=radiologist.id)
                ))
                .then(lambda: radiologist.active_study == study.id)
                .then(lambda: radiologist.estimate_reading_time(study) < timedelta(hours=2)))
    
    @staticmethod
    def fatigue_management():
        return (BehaviorSpecification("Radiologist fatigue affects reading time")
                .given(
                    radiologist=Radiologist("R001", "Dr. Smith", ["MD"], ["MRI"]),
                    initial_fatigue=1.0
                )
                .when(lambda: [radiologist.update_fatigue() for _ in range(20)])
                .then(lambda: radiologist.fatigue_factor < initial_fatigue)
                .then(lambda: radiologist.estimate_reading_time(Study("S001", "P001", "MRI")) 
                      > timedelta(minutes=20)))

# Nurse Behavior Specifications
class NurseBehaviors:
    @staticmethod
    def patient_preparation():
        return (BehaviorSpecification("Nurse prepares patient for MRI with contrast")
                .given(
                    nurse=Nurse("N001", "Jane Doe", ["RN", "BSN"], "Radiology"),
                    patient=Patient("P001", "MRN123"),
                    study=Study("S001", "P001", "MRI"),
                    requires_contrast=True
                )
                .when(lambda: nurse.prepare_patient_for_study(patient, study))
                .then(lambda: len(nurse.task_queue) > 0)
                .then(lambda: "verify_contrast_consent" in [task.name for task in nurse.task_queue]))

# Urologist Behavior Specifications
class UrologistBehaviors:
    @staticmethod
    def treatment_decision():
        return (BehaviorSpecification("Urologist makes treatment decision for high-risk patient")
                .given(
                    urologist=Urologist("U001", "Dr. Johnson", ["MD", "FACS"]),
                    staging_report=StagingReport(
                        patient_id="P001",
                        stage="T3a",
                        gleason_score=8,
                        psa=15.5,
                        risk_group="High"
                    )
                )
                .when(lambda: urologist.review_staging_report(staging_report))
                .then(lambda: decision.urgent_consultation == True)
                .then(lambda: decision.follow_up_interval <= timedelta(days=7)))

# ============================================
# Workflow Orchestration
# ============================================

class ClinicalWorkflow:
    """Orchestrates clinical workflows across stakeholders"""
    def __init__(self):
        self.event_store = EventStore()
        self.saga_manager = SagaManager()
        
    async def patient_journey(self, patient: Patient, initial_complaint: str):
        """Orchestrate complete patient journey"""
        # 1. Patient admission
        admission_event = PatientAdmitted(
            patient_id=patient.id,
            chief_complaint=initial_complaint
        )
        self.event_store.append(admission_event)
        
        # 2. Urologist orders study
        urologist = self.find_available_urologist()
        study_order = StudyOrdered(
            patient_id=patient.id,
            ordered_by=urologist.id,
            modality="MRI with contrast",
            clinical_indication=initial_complaint
        )
        self.event_store.append(study_order)
        
        # 3. Nurse prepares patient
        nurse = self.find_available_nurse("Radiology")
        preparation_saga = PatientPreparationSaga(patient, nurse)
        await self.saga_manager.execute(preparation_saga)
        
        # 4. Technologist acquires images
        tech = self.find_available_technologist("MRI")
        acquisition_saga = ImageAcquisitionSaga(study_order.study_id, tech)
        await self.saga_manager.execute(acquisition_saga)
        
        # 5. AI preliminary analysis
        ai_saga = AIAnalysisSaga(study_order.study_id)
        await self.saga_manager.execute(ai_saga)
        
        # 6. Radiologist interpretation
        radiologist = self.find_best_radiologist(study_order)
        interpretation_saga = RadiologyInterpretationSaga(study_order.study_id, radiologist)
        await self.saga_manager.execute(interpretation_saga)
        
        # 7. Report to referring physician
        report_delivery_saga = ReportDeliverySaga(study_order.study_id, urologist)
        await self.saga_manager.execute(report_delivery_saga)
        
        # 8. Treatment planning
        treatment_saga = TreatmentPlanningSaga(patient, urologist)
        await self.saga_manager.execute(treatment_saga)

class SagaManager:
    """Manages long-running transactions"""
    def __init__(self):
        self.active_sagas = {}
        
    async def execute(self, saga: 'Saga'):
        """Execute a saga with compensation logic"""
        saga_id = str(uuid.uuid4())
        self.active_sagas[saga_id] = saga
        
        try:
            await saga.execute()
            self.active_sagas.pop(saga_id)
        except Exception as e:
            # Execute compensation logic
            await saga.compensate()
            self.active_sagas.pop(saga_id)
            raise