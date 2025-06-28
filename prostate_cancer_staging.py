# Enterprise Prostate Cancer Staging System
# Real-time Multi-Hospital Deployment with Clinical Interfaces

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pandas as pd
import pydicom
import cv2
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandShiftIntensityd
)
from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import mlflow
import optuna
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import asyncio
import aiohttp
from datetime import datetime
import redis
import kafka
from kafka import KafkaProducer, KafkaConsumer
import grpc
import boto3
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import jwt
from cryptography.fernet import Fernet
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ray
from ray import serve
import onnx
import onnxruntime as ort
import tensorrt as trt
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize monitoring
sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[SqlalchemyIntegration()]
)

# Prometheus metrics
prediction_counter = Counter('prostate_staging_predictions_total', 'Total predictions made')
prediction_latency = Histogram('prostate_staging_prediction_duration_seconds', 'Prediction latency')
active_connections = Gauge('prostate_staging_active_connections', 'Active WebSocket connections')

# ============================================
# Database Schema for Multi-Hospital System
# ============================================

Base = declarative_base()

class Hospital(Base):
    __tablename__ = 'hospitals'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    location = Column(String)
    api_key = Column(String, unique=True)
    encryption_key = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON)

class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    hospital_id = Column(String, nullable=False)
    mrn = Column(String)  # Medical Record Number (encrypted)
    age = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

class Study(Base):
    __tablename__ = 'studies'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, nullable=False)
    study_date = Column(DateTime)
    modality = Column(String)  # MRI, CT, Pathology
    status = Column(String)  # pending, processing, completed, failed
    priority = Column(Integer, default=5)  # 1-10, 1 being highest
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    results = Column(JSON)
    metadata = Column(JSON)

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    study_id = Column(String, nullable=False)
    model_version = Column(String)
    t_stage = Column(String)
    n_stage = Column(String)
    m_stage = Column(String)
    gleason_score = Column(Integer)
    psa_level = Column(Float)
    risk_group = Column(String)
    confidence_score = Column(Float)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_by = Column(String)
    reviewed_at = Column(DateTime)
    review_notes = Column(String)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String)
    action = Column(String)
    resource_type = Column(String)
    resource_id = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON)

# ============================================
# Real-time Processing Infrastructure
# ============================================

class MessageQueue:
    """Kafka-based message queue for real-time processing"""
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.consumer = None
    
    def send_study(self, study_data: Dict):
        """Send study for processing"""
        self.producer.send('prostate-studies', study_data)
        self.producer.flush()
    
    def consume_studies(self, topic='prostate-studies'):
        """Consume studies for processing"""
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='prostate-staging-group'
        )
        return self.consumer

class CacheManager:
    """Redis-based caching for fast access"""
    def __init__(self, host='localhost', port=6379):
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = 3600  # 1 hour default TTL
    
    def get(self, key: str) -> Optional[Dict]:
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def set(self, key: str, value: Dict, ttl: Optional[int] = None):
        self.redis_client.setex(
            key, 
            ttl or self.ttl, 
            json.dumps(value)
        )
    
    def invalidate(self, pattern: str):
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)

# ============================================
# Optimized Model Serving
# ============================================

@ray.remote
class ModelServer:
    """Ray Serve for distributed model serving"""
    def __init__(self, model_path: str, use_tensorrt: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tensorrt = use_tensorrt and torch.cuda.is_available()
        
        if self.use_tensorrt:
            self.model = self._load_tensorrt_model(model_path)
        else:
            self.model = self._load_torch_model(model_path)
    
    def _load_tensorrt_model(self, model_path: str):
        """Load TensorRT optimized model"""
        logger.info("Loading TensorRT model...")
        # TensorRT optimization code here
        return None  # Placeholder
    
    def _load_torch_model(self, model_path: str):
        """Load PyTorch model"""
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model
    
    async def predict(self, data: Dict) -> Dict:
        """Make prediction with optimized inference"""
        with prediction_latency.time():
            # Preprocess data
            processed_data = self._preprocess(data)
            
            # Run inference
            with torch.no_grad():
                if self.use_tensorrt:
                    output = self._tensorrt_inference(processed_data)
                else:
                    output = self.model(processed_data)
            
            # Postprocess results
            results = self._postprocess(output)
            
            prediction_counter.inc()
            return results
    
    def _preprocess(self, data: Dict) -> torch.Tensor:
        """Preprocess input data"""
        # Implementation here
        return torch.randn(1, 3, 256, 256)  # Placeholder
    
    def _postprocess(self, output: torch.Tensor) -> Dict:
        """Postprocess model output"""
        # Implementation here
        return {"stage": "T2a", "confidence": 0.95}  # Placeholder

# ============================================
# Clinical User Interface API
# ============================================

app = FastAPI(title="Prostate Cancer Staging API", version="2.0")

# CORS middleware for hospital systems
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific hospital domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class StudyRequest(BaseModel):
    patient_id: str
    modality: str
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = {}

class PredictionResponse(BaseModel):
    study_id: str
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    visualizations: Dict[str, str]  # Base64 encoded images

class ClinicalReport(BaseModel):
    patient_id: str
    study_date: datetime
    findings: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]

# Authentication middleware
async def verify_hospital_token(token: str) -> Optional[str]:
    """Verify hospital API token"""
    try:
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        return payload.get("hospital_id")
    except:
        return None

@app.post("/api/v1/studies/upload")
async def upload_study(
    files: List[UploadFile] = File(...),
    request: StudyRequest = None,
    token: str = None
):
    """Upload medical images for processing"""
    hospital_id = await verify_hospital_token(token)
    if not hospital_id:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    # Create study record
    study_id = str(uuid.uuid4())
    
    # Queue for processing
    queue = MessageQueue()
    queue.send_study({
        "study_id": study_id,
        "hospital_id": hospital_id,
        "patient_id": request.patient_id,
        "modality": request.modality,
        "priority": request.priority,
        "files": [file.filename for file in files]
    })
    
    return {"study_id": study_id, "status": "queued"}

@app.get("/api/v1/studies/{study_id}/results")
async def get_results(study_id: str, token: str = None):
    """Get study results"""
    hospital_id = await verify_hospital_token(token)
    if not hospital_id:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    # Check cache first
    cache = CacheManager()
    cached_results = cache.get(f"results:{study_id}")
    if cached_results:
        return cached_results
    
    # Query database
    # Implementation here
    
    return PredictionResponse(
        study_id=study_id,
        predictions={
            "t_stage": "T2a",
            "n_stage": "N0",
            "m_stage": "M0",
            "gleason_score": 7,
            "risk_group": "intermediate"
        },
        confidence_scores={
            "t_stage": 0.92,
            "n_stage": 0.88,
            "m_stage": 0.95
        },
        processing_time=2.3,
        visualizations={}
    )

@app.websocket("/ws/studies/{study_id}")
async def websocket_endpoint(websocket: WebSocket, study_id: str):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.inc()
    
    try:
        while True:
            # Send real-time updates
            data = await get_study_updates(study_id)
            await websocket.send_json(data)
            await asyncio.sleep(1)
    except:
        active_connections.dec()
        await websocket.close()

# ============================================
# Doctor/Radiologist Dashboard
# ============================================

def create_clinical_dashboard():
    """Streamlit dashboard for medical professionals"""
    st.set_page_config(
        page_title="Prostate Cancer Staging Dashboard",
        page_icon="🏥",
        layout="wide"
    )
    
    # Custom CSS for medical UI
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .urgent {
        background-color: #ff4b4b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🏥 Prostate Cancer Staging System")
    with col2:
        st.metric("Active Studies", "12", "+3")
    with col3:
        if st.button("🔔 Notifications"):
            st.info("3 studies require review")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "New Study", "Patient Search", "Reports", "Settings"]
        )
        
        st.divider()
        
        # Quick stats
        st.subheader("Today's Statistics")
        st.metric("Studies Processed", "47")
        st.metric("Average Processing Time", "2.3 min")
        st.metric("Accuracy Rate", "96.8%")
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "New Study":
        show_new_study_form()
    elif page == "Patient Search":
        show_patient_search()
    elif page == "Reports":
        show_reports()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Main dashboard view"""
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Urgent Cases", "📈 Analytics", "🔄 Processing Queue"])
    
    with tab1:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Patients", "1,234", "↑ 12%")
            st.caption("This month")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("High Risk Cases", "23", "↑ 3")
            st.caption("Requiring immediate attention")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Turnaround", "18 min", "↓ 5 min")
            st.caption("From upload to results")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Accuracy", "97.2%", "↑ 0.3%")
            st.caption("Last 30 days")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent studies table
        st.subheader("Recent Studies")
        recent_studies = pd.DataFrame({
            'Patient ID': ['P-1234', 'P-1235', 'P-1236', 'P-1237', 'P-1238'],
            'Study Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
            'Modality': ['MRI', 'MRI + Path', 'MRI', 'Path', 'MRI + Path'],
            'Stage': ['T2a N0 M0', 'T3b N1 M0', 'T1c N0 M0', 'T2b N0 M0', 'T3a N0 M0'],
            'Risk': ['Intermediate', 'High', 'Low', 'Intermediate', 'High'],
            'Status': ['Reviewed', 'Pending Review', 'Reviewed', 'Processing', 'Pending Review']
        })
        
        # Style the dataframe
        def highlight_risk(val):
            if val == 'High':
                return 'background-color: #ff4b4b; color: white'
            elif val == 'Intermediate':
                return 'background-color: #ffa726; color: white'
            else:
                return 'background-color: #66bb6a; color: white'
        
        styled_df = recent_studies.style.applymap(highlight_risk, subset=['Risk'])
        st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        # Urgent cases
        st.subheader("🚨 Cases Requiring Immediate Attention")
        
        urgent_cases = pd.DataFrame({
            'Patient ID': ['P-1235', 'P-1238', 'P-1241'],
            'Age': [68, 72, 65],
            'PSA': [15.2, 22.1, 18.5],
            'Gleason': ['4+4', '4+5', '4+3'],
            'Stage': ['T3b N1 M0', 'T3a N0 M0', 'T3a N1 M0'],
            'Days Waiting': [2, 1, 3]
        })
        
        for idx, row in urgent_cases.iterrows():
            with st.expander(f"Patient {row['Patient ID']} - {row['Days Waiting']} days waiting"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Age:** {row['Age']} years")
                    st.write(f"**PSA Level:** {row['PSA']} ng/mL")
                    st.write(f"**Gleason Score:** {row['Gleason']}")
                with col2:
                    st.write(f"**Clinical Stage:** {row['Stage']}")
                    if st.button(f"Review Case", key=f"review_{idx}"):
                        st.success("Opening case review...")
    
    with tab3:
        # Analytics
        st.subheader("📈 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Staging distribution
            fig_stages = go.Figure(data=[
                go.Bar(name='T1', x=['T1a', 'T1b', 'T1c'], y=[12, 19, 28]),
                go.Bar(name='T2', x=['T2a', 'T2b', 'T2c'], y=[31, 27, 18]),
                go.Bar(name='T3', x=['T3a', 'T3b'], y=[15, 8]),
                go.Bar(name='T4', x=['T4'], y=[5])
            ])
            fig_stages.update_layout(
                title="Stage Distribution (Last 30 Days)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_stages, use_container_width=True)
        
        with col2:
            # Risk group pie chart
            fig_risk = go.Figure(data=[go.Pie(
                labels=['Low Risk', 'Intermediate Risk', 'High Risk', 'Very High Risk'],
                values=[145, 234, 121, 45],
                hole=.3
            )])
            fig_risk.update_layout(
                title="Risk Group Distribution",
                height=400
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Processing time trend
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        times = np.random.normal(18, 3, 30)
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=dates, y=times,
            mode='lines+markers',
            name='Processing Time',
            line=dict(color='#0066cc', width=2)
        ))
        fig_time.update_layout(
            title="Average Processing Time Trend",
            xaxis_title="Date",
            yaxis_title="Time (minutes)",
            height=300
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with tab4:
        # Processing queue
        st.subheader("🔄 Current Processing Queue")
        
        queue_data = pd.DataFrame({
            'Study ID': ['S-5678', 'S-5679', 'S-5680', 'S-5681'],
            'Patient ID': ['P-1240', 'P-1241', 'P-1242', 'P-1243'],
            'Modality': ['MRI', 'MRI + Path', 'Path', 'MRI'],
            'Priority': ['High', 'Urgent', 'Normal', 'High'],
            'Progress': [75, 45, 20, 60],
            'ETA': ['2 min', '5 min', '8 min', '3 min']
        })
        
        for idx, row in queue_data.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            
            with col1:
                st.write(f"**{row['Study ID']}** - Patient {row['Patient ID']}")
            
            with col2:
                if row['Priority'] == 'Urgent':
                    st.markdown(f"<span class='urgent'>URGENT</span>", unsafe_allow_html=True)
                else:
                    st.write(row['Priority'])
            
            with col3:
                st.progress(row['Progress'] / 100)
            
            with col4:
                st.write(f"ETA: {row['ETA']}")

def show_new_study_form():
    """Form for uploading new study"""
    st.header("📤 Upload New Study")
    
    with st.form("new_study_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID")
            age = st.number_input("Age", min_value=0, max_value=120)
            psa = st.number_input("PSA Level (ng/mL)", min_value=0.0, step=0.1)
        
        with col2:
            modality = st.selectbox("Modality", ["MRI", "Pathology", "MRI + Pathology"])
            priority = st.select_slider("Priority", options=range(1, 11), value=5)
            clinical_notes = st.text_area("Clinical Notes")
        
        st.subheader("Upload Files")
        
        if modality in ["MRI", "MRI + Pathology"]:
            mri_files = st.file_uploader(
                "MRI Images (DICOM/NIfTI)", 
                accept_multiple_files=True,
                type=['dcm', 'nii', 'nii.gz']
            )
        
        if modality in ["Pathology", "MRI + Pathology"]:
            path_files = st.file_uploader(
                "Pathology Slides", 
                accept_multiple_files=True,
                type=['png', 'jpg', 'tiff', 'svs']
            )
        
        submitted = st.form_submit_button("Submit for Processing", type="primary")
        
        if submitted:
            with st.spinner("Uploading and queuing study..."):
                # Process upload
                study_id = str(uuid.uuid4())
                st.success(f"✅ Study uploaded successfully! Study ID: {study_id}")
                st.info("You will receive a notification when processing is complete.")

def show_patient_search():
    """Patient search interface"""
    st.header("🔍 Patient Search")
    
    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
    
    with search_col1:
        search_term = st.text_input("Search by Patient ID, Name, or MRN")
    
    with search_col2:
        date_range = st.date_input("Date Range", value=[])
    
    with search_col3:
        if st.button("Search", type="primary"):
            # Perform search
            st.success("Search completed")
    
    # Search results
    if search_term:
        results = pd.DataFrame({
            'Patient ID': ['P-1234', 'P-1235', 'P-1236'],
            'Name': ['John Doe', 'James Smith', 'Robert Johnson'],
            'Age': [65, 72, 68],
            'Last Study': ['2024-01-15', '2024-01-14', '2024-01-13'],
            'Stage': ['T2a N0 M0', 'T3b N1 M0', 'T1c N0 M0'],
            'Risk': ['Intermediate', 'High', 'Low']
        })
        
        st.dataframe(results, use_container_width=True)
        
        # Patient details
        selected_patient = st.selectbox("Select patient for details", results['Patient ID'])
        
        if selected_patient:
            show_patient_details(selected_patient)

def show_patient_details(patient_id):
    """Show detailed patient information"""
    st.subheader(f"Patient Details: {patient_id}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Studies", "Trends", "Reports"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demographics**")
            st.write("Age: 65 years")
            st.write("MRN: XXXXX1234")
            st.write("First Visit: 2023-06-15")
            
            st.markdown("**Current Status**")
            st.write("Stage: T2a N0 M0")
            st.write("Gleason Score: 3+4=7")
            st.write("Risk Group: Intermediate")
        
        with col2:
            st.markdown("**Recent Values**")
            st.write("PSA: 8.5 ng/mL")
            st.write("Prostate Volume: 45 cc")
            st.write("Positive Cores: 3/12")
            
            st.markdown("**Treatment**")
            st.write("Current: Active Surveillance")
            st.write("Started: 2023-07-01")
    
    with tab2:
        # Studies timeline
        studies_df = pd.DataFrame({
            'Date': pd.date_range(start='2023-06-15', periods=5, freq='3M'),
            'Type': ['MRI', 'Biopsy', 'MRI', 'MRI + Path', 'MRI'],
            'PSA': [6.2, 7.1, 7.8, 8.2, 8.5],
            'Stage': ['T1c', 'T2a', 'T2a', 'T2a', 'T2a']
        })
        
        st.dataframe(studies_df, use_container_width=True)
    
    with tab3:
        # PSA trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=studies_df['Date'],
            y=studies_df['PSA'],
            mode='lines+markers',
            name='PSA Level',
            line=dict(color='#ff6b6b', width=3)
        ))
        fig.update_layout(
            title="PSA Trend",
            xaxis_title="Date",
            yaxis_title="PSA (ng/mL)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_reports():
    """Reports generation interface"""
    st.header("📊 Generate Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Patient Summary", "Department Statistics", "Quality Metrics", "Research Export"]
    )
    
    if report_type == "Department Statistics":
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
        