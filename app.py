"""
ModelForge - Industrial Forge Theme
End-to-end synthetic data generation and model fine-tuning platform
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import json
import asyncio
import time
import os

from src.backboard_manager import BackboardManager, run_async
from src.generator import DataGenerator
from src.model_registry import (
    BackboardModelRegistry, 
    FALLBACK_MODELS,
    initialize_registry,
    get_registry
)
from src.finetuning_manager import (
    DockerFineTuningManager,
    TrainingConfig,
    UNSLOTH_MODELS
)


# ==================== INDUSTRIAL FORGE THEME ====================

def load_forge_theme():
    """Load industrial forge theme CSS"""
    st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background-color: #0b0c0d;
        color: #a0a0a0;
        font-family: 'Rajdhani', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border: 1px solid #1a1a1a;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #f96124;
    }
    
    /* Industrial plate styling */
    .industrial-plate {
        background: #141414;
        border: 1px solid #333;
        border-left: 4px solid #f96124;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.5);
        position: relative;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Technical grid background */
    .industrial-plate::before {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 20px;
        height: 20px;
        background: 
            linear-gradient(to bottom, #222 1px, transparent 1px),
            linear-gradient(to right, #222 1px, transparent 1px);
        background-size: 4px 4px;
        opacity: 0.3;
    }
    
    /* Primary color (forge orange) */
    .text-primary {
        color: #f96124 !important;
    }
    .bg-primary {
        background-color: #f96124 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: #1a1a1a;
        border: 1px solid #333;
        color: #888;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.7rem;
        border-radius: 0px; /* Sharp edges */
        transition: all 0.1s;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: #252525;
        color: #fff;
        border-color: #f96124;
        box-shadow: 0 0 8px rgba(249, 97, 36, 0.2);
    }
    .stButton > button:active {
        background: #f96124;
        color: #000;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #f96124;
        border: 1px solid #f96124;
        color: #000;
    }
    .stButton > button[kind="primary"]:hover {
        background: #ff7b42;
        box-shadow: 0 0 15px rgba(249, 97, 36, 0.4);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #080808;
        border: 1px solid #333;
        color: #ccc;
        border-radius: 0px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #f96124;
        color: white;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #080808;
        border: 1px solid #333;
        color: #ccc;
        border-radius: 0px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Pipeline card */
    .pipeline-card {
        background: #111;
        border: 1px solid #222;
        border-radius: 0px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 2px solid #333;
    }
    .pipeline-card:hover {
        border-left: 2px solid #f96124;
        background: #161616;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        background: #000;
        border: 1px solid #333;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .status-online { color: #22c55e; border-color: #22c55e; }
    .status-running { color: #f96124; border-color: #f96124; }
    .status-complete { color: #3b82f6; border-color: #3b82f6; }
    .status-idle { color: #666; border-color: #444; }
    
    /* Terminal/Log styling */
    .terminal-log {
        background: #050505;
        border: 1px solid #333;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #bbb;
        max-height: 400px;
        overflow-y: auto;
    }
    .log-time { color: #555; margin-right: 0.5rem; }
    .log-level-info { color: #3b82f6; }
    .log-level-warn { color: #eab308; }
    .log-level-error { color: #ef4444; }
    .log-level-success { color: #22c55e; }
    
    /* Model card in selector */
    .model-card-compact {
        background: #111;
        border: 1px solid #222;
        padding: 0.5rem;
        margin-bottom: 0.25rem;
        transition: all 0.2s;
        cursor: pointer;
    }
    .model-card-compact:hover {
        border-color: #f96124;
        background: #1a1a1a;
    }
    
    /* Stat card */
    .stat-card {
        background: #111;
        border: 1px solid #222;
        padding: 1rem;
        position: relative;
    }
    .stat-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: #333;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 900;
        color: white;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #8c6b5d;
    }
    
    /* Progress bar */
    .progress-bar-container {
        background: #0f0907;
        border: 1px solid #38302a;
        border-radius: 4px;
        height: 12px;
        overflow: hidden;
        position: relative;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #7c2d12 0%, #ff4d00 50%, #fbbf24 100%);
        height: 100%;
        box-shadow: 0 0 10px rgba(255, 77, 0, 0.5);
        transition: width 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== SESSION STATE MANAGEMENT ====================

def init_session_state():
    """Initialize session state variables"""
    if 'view' not in st.session_state:
        st.session_state.view = 'setup'  # setup, dashboard, data_generation, data_viewer, fine_tuning
    
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    
    if 'pipelines' not in st.session_state:
        st.session_state.pipelines = []
    
    if 'active_pipeline' not in st.session_state:
        st.session_state.active_pipeline = None
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    
    if 'show_model_selector' not in st.session_state:
        st.session_state.show_model_selector = False
    
    if 'available_models' not in st.session_state:
        st.session_state.available_models = FALLBACK_MODELS
    
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    # Dataset management
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}  # {dataset_id: Dataset}
    
    if 'active_dataset' not in st.session_state:
        st.session_state.active_dataset = None
    
    # Document cache for RAG add-on (shared across modes)
    if 'global_documents' not in st.session_state:
        st.session_state.global_documents = []  # [{id, name, content}]
    
    # Shared BackboardManager for URL fetching (created lazily when API key is set)
    if 'shared_backboard_manager' not in st.session_state:
        st.session_state.shared_backboard_manager = None
    
    # Fine-tuning state
    if 'finetuning_manager' not in st.session_state:
        st.session_state.finetuning_manager = None
    
    if 'finetuning_status' not in st.session_state:
        st.session_state.finetuning_status = None  # 'training', 'complete', 'error'
    
    if 'finetuning_progress' not in st.session_state:
        st.session_state.finetuning_progress = {}
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = []  # List of saved model paths


# ==================== DATASET MANAGEMENT ====================

@dataclass
class Dataset:
    """Represents a dataset that can be appended to"""
    id: str
    name: str
    mode: int
    mode_name: str
    file_path: str
    samples: List[Dict]
    created_at: datetime
    thread_id: Optional[str] = None  # For memory continuity
    assistant_id: Optional[str] = None
    document_ids: List[str] = None  # RAG documents
    
    def __post_init__(self):
        if self.document_ids is None:
            self.document_ids = []
    
    @property
    def sample_count(self) -> int:
        return len(self.samples)
    
    def add_samples(self, new_samples):
        """Add new samples to the dataset. Accepts both dicts and GeneratedSample objects.
        Filters out empty/invalid samples automatically."""
        added_count = 0
        for sample in new_samples:
            if hasattr(sample, 'to_alpaca'):
                data = sample.to_alpaca()
            else:
                data = sample
            
            # Validate sample has content
            instruction = data.get('instruction', '').strip()
            output = data.get('output', '').strip()
            if instruction or output:  # Must have at least instruction or output
                self.samples.append(data)
                added_count += 1
        
        if added_count < len(new_samples):
            print(f"Warning: Filtered out {len(new_samples) - added_count} empty samples")
        
        self.save()
    
    def save(self):
        """Save dataset to JSONL file and metadata to companion JSON file"""
        # Save samples to JSONL
        with open(self.file_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Save metadata to companion file
        meta_path = self.file_path.replace('.jsonl', '.meta.json')
        metadata = {
            'id': str(self.id),
            'name': self.name,
            'mode': self.mode,
            'mode_name': self.mode_name,
            'thread_id': str(self.thread_id) if self.thread_id else None,
            'assistant_id': str(self.assistant_id) if self.assistant_id else None,
            'document_ids': self.document_ids,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'Dataset':
        """Load dataset from existing JSONL file and metadata from companion JSON"""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Try to load metadata from companion file
        meta_path = file_path.replace('.jsonl', '.meta.json')
        metadata = {}
        if Path(meta_path).exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")

        # Extract info from filename as fallback
        filename = Path(file_path).stem
        parts = filename.split('_', 1)
        dataset_id = metadata.get('id') or (parts[0] if len(parts) > 1 else filename[:8])
        name = metadata.get('name') or (parts[1].replace('_', ' ') if len(parts) > 1 else filename)

        # Parse created_at from metadata
        created_at = None
        if metadata.get('created_at'):
            try:
                created_at = datetime.fromisoformat(metadata['created_at'])
            except:
                pass
        if not created_at:
            created_at = datetime.fromtimestamp(Path(file_path).stat().st_mtime)

        return cls(
            id=dataset_id,
            name=name,
            mode=metadata.get('mode', 1),
            mode_name=metadata.get('mode_name', "Imported Dataset"),
            file_path=file_path,
            samples=samples,
            created_at=created_at,
            thread_id=metadata.get('thread_id'),
            assistant_id=metadata.get('assistant_id'),
            document_ids=metadata.get('document_ids', [])
        )


def load_existing_datasets():
    """Load all existing datasets from data/ folder"""
    data_dir = Path("data")
    if not data_dir.exists():
        return
    
    for jsonl_file in data_dir.glob("*.jsonl"):
        dataset_id = jsonl_file.stem.split('_')[0]
        if dataset_id not in st.session_state.datasets:
            try:
                dataset = Dataset.load_from_file(str(jsonl_file))
                st.session_state.datasets[dataset.id] = dataset
            except Exception as e:
                print(f"Error loading {jsonl_file}: {e}")


# ==================== PIPELINE MANAGEMENT ====================

class Pipeline:
    """Represents a data generation pipeline"""
    def __init__(self, name: str, pipeline_id: str):
        self.id = pipeline_id
        self.name = name
        self.created_at = datetime.now()
        self.status = "idle"  # idle, running, complete, error
        self.samples_generated = 0
        self.target_samples = 0
        self.model = None
        self.mode = 1
        self.mode_name = "Mode 1: General Chat (Memory-Driven)"
        self.logs = []
        
        # Dataset linking for append operations
        self.dataset_id = None  # Link to existing dataset for appending
        self.thread_id = None   # For memory continuity
        self.assistant_id = None
        
        # RAG Add-on (available for all modes)
        self.use_rag = False
        self.rag_documents = []  # [{id, name, content}]
        
        # Mode-specific configurations
        self.config = {
            "topic": "",
            "style": "",
            "document_ids": [],
            "code_language": "python",
            "tools": [],
            "uploaded_files": []
        }
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'samples_generated': self.samples_generated,
            'target_samples': self.target_samples,
            'model': self.model,
            'mode': self.mode,
            'mode_name': self.mode_name,
            'dataset_id': self.dataset_id,
            'use_rag': self.use_rag
        }


def create_pipeline(name: str, dataset_id: str = None):
    """Create a new pipeline, optionally linked to existing dataset"""
    import uuid
    pipeline_id = str(uuid.uuid4())[:8]
    pipeline = Pipeline(name, pipeline_id)
    
    # Link to existing dataset if provided
    if dataset_id and dataset_id in st.session_state.datasets:
        dataset = st.session_state.datasets[dataset_id]
        pipeline.dataset_id = dataset_id
        pipeline.thread_id = dataset.thread_id
        pipeline.assistant_id = dataset.assistant_id
        pipeline.mode = dataset.mode
        pipeline.mode_name = dataset.mode_name
        pipeline.rag_documents = [{'id': doc_id} for doc_id in dataset.document_ids]
    
    st.session_state.pipelines.append(pipeline)
    st.session_state.active_pipeline = pipeline
    st.session_state.view = 'data_generation'


def add_log(pipeline: Pipeline, level: str, message: str):
    """Add a log entry to the pipeline"""
    pipeline.logs.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'level': level,
        'message': message
    })


# ==================== VIEW: SETUP MODAL ====================

def render_setup_view():
    """Render initial setup modal for name and API key"""
    # Center the content vertically and horizontally
    _, col_center, _ = st.columns([1, 1.5, 1])
    
    with col_center:
        st.write("") # Spacer
        st.write("") # Spacer
        
        # Logo Image
        lc1, lc2, lc3 = st.columns([3, 1, 3])
        with lc2:
            try:
                st.image("logo.png", use_container_width=True)
            except:
                pass

        # Logo and Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #f96124; font-size: 3.5rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em; line-height: 1; text-shadow: 0 0 20px rgba(249, 97, 36, 0.3);">
                MODEL<br>FORGE
            </h1>
            <div style="background: linear-gradient(90deg, transparent, #f96124, transparent); height: 2px; width: 100%; margin: 1.5rem auto; opacity: 0.5;"></div>
            <p style="color: #8c6b5d; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.3em;">
                // AUTHENTICATION_PROTOCOL //
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login Form
        with st.form("login_form", border=True):
            st.markdown("### [ACCESS_CONTROL]")
            
            name = st.text_input(
                "OPERATOR_ID",
                placeholder="Enter designation...",
                help="Enter your username to identify sessions"
            )
            
            api_key = st.text_input(
                "SECURITY_KEY",
                type="password",
                placeholder="sk-...",
                help="Enter your Backboard API Key"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button(
                "[INITIALIZE_SYSTEM]", 
                type="primary", 
                use_container_width=True
            )
            
            if submitted:
                if name and api_key:
                    st.session_state.user_name = name
                    st.session_state.api_key = api_key
                    st.session_state.view = 'dashboard'
                    st.rerun()
                else:
                    st.error("[ACCESS_DENIED] Credentials Incomplete")


# ==================== VIEW: DASHBOARD ====================

def render_dashboard():
    """Render main dashboard with pipelines"""
    # Load existing datasets
    load_existing_datasets()
    
    # Auto-load models if API key is available and models not yet loaded
    if st.session_state.api_key and not st.session_state.models_loaded:
        try:
            import asyncio
            registry = initialize_registry(st.session_state.api_key)
            models = asyncio.run(registry.fetch_models())
            st.session_state.available_models = models
            st.session_state.models_loaded = True
        except Exception as e:
            print(f"Failed to auto-load models: {e}")
    
    # Header
    col1, col2, col3 = st.columns([4, 1, 3])
    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                SYSTEM DASHBOARD
            </h1>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span class="status-badge status-online">
                    <span style="width: 6px; height: 6px; background: #22c55e; display: inline-block;"></span>
                    SYSTEM ONLINE
                </span>
                <span style="color: #666; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                    OPERATOR: {st.session_state.user_name.upper()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("[VIEW_DATA]", use_container_width=True):
            st.session_state.view = 'data_viewer'
            st.rerun()
    
    with col3:
        col3a, col3b, col3c, col3d = st.columns(4)
        with col3a:
            if st.button("[TRAIN]", use_container_width=True):
                st.session_state.view = 'fine_tuning'
                st.rerun()
        with col3b:
            if st.button("[TEST]", use_container_width=True):
                st.session_state.view = 'inference'
                st.rerun()
        with col3c:
            if st.button("🤗 DEPLOY", use_container_width=True):
                st.session_state.view = 'hf_deploy'
                st.rerun()
        with col3d:
            if st.button("[NEW]", type="primary", use_container_width=True):
                st.session_state.show_create_pipeline = True
                st.rerun()
    
    # Stats cards
    st.markdown("### SYSTEM METRICS")
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Pipelines</div>
            <div class="stat-value">{len(st.session_state.pipelines)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[1]:
        running = len([p for p in st.session_state.pipelines if p.status == "running"])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Running</div>
            <div class="stat-value" style="color: #f96124;">{running}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[2]:
        complete = len([p for p in st.session_state.pipelines if p.status == "complete"])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Complete</div>
            <div class="stat-value" style="color: #22c55e;">{complete}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[3]:
        total_samples = sum(p.samples_generated for p in st.session_state.pipelines)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Samples</div>
            <div class="stat-value">{total_samples}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Pipeline list
    st.markdown("### ACTIVE PROCESSES")
    
    if not st.session_state.pipelines:
        st.info("[INFO] No pipelines initialized. Create a new pipeline to begin.")
    else:
        for pipeline in st.session_state.pipelines:
            render_pipeline_card(pipeline)
    
    # Create pipeline modal
    if st.session_state.get('show_create_pipeline', False):
        render_create_pipeline_modal()


def render_pipeline_card(pipeline: Pipeline):
    """Render a pipeline card"""
    status_class = f"status-{pipeline.status}"
    status_text = pipeline.status.upper()
    
    # Calculate progress
    progress = 0
    if pipeline.target_samples > 0:
        progress = (pipeline.samples_generated / pipeline.target_samples) * 100
    
    with st.container():
        col1, col2, col3 = st.columns([5, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="pipeline-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <div>
                        <h3 style="color: white; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;">
                            {pipeline.name}
                        </h3>
                        <span class="status-badge {status_class}">{status_text}</span>
                    </div>
                    <span style="color: #666; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;">
                        ID: {pipeline.id}
                    </span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">
                        <span>Progress</span>
                        <span>{pipeline.samples_generated} / {pipeline.target_samples} samples</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {progress}%;"></div>
                    </div>
                </div>
                <div style="display: flex; gap: 1rem; font-size: 0.7rem; color: #666;">
                    <span>DATE: {pipeline.created_at.strftime('%Y-%m-%d %H:%M')}</span>
                    {f'<span>MODEL: {pipeline.model}</span>' if pipeline.model else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("Open", key=f"open_{pipeline.id}", use_container_width=True):
                st.session_state.active_pipeline = pipeline
                st.session_state.view = 'data_generation'
                st.rerun()
        
        with col3:
            if st.button("Delete", key=f"delete_{pipeline.id}", use_container_width=True):
                st.session_state.pipelines.remove(pipeline)
                st.rerun()


def render_create_pipeline_modal():
    """Render create pipeline modal"""
    with st.container():
        st.markdown("---")
        st.markdown("### [CREATE_NEW_PIPELINE]")
        st.markdown("Name your data generation pipeline and select the generation mode.")
        
        pipeline_name = st.text_input(
            "Pipeline Name",
            placeholder="e.g., Customer Support Training Data",
            key="new_pipeline_name"
        )
        
        # Mode selection
        mode = st.selectbox(
            "Generation Mode",
            [
                "Mode 1: General Chat (Memory-Driven)",
                "Mode 2: Knowledge Injection (RAG)",
                "Mode 3: Code Specialist",
                "Mode 4: Agent / Tool Use",
                "Mode 5: Reasoning (CoT)"
            ],
            key="new_pipeline_mode"
        )
        
        # Mode descriptions
        mode_descriptions = {
            "Mode 1: General Chat (Memory-Driven)": "Generate unique conversational Q&A using memory deduplication",
            "Mode 2: Knowledge Injection (RAG)": "Generate data grounded in uploaded documents",
            "Mode 3: Code Specialist": "Generate code challenges and solutions with specialized models",
            "Mode 4: Agent / Tool Use": "Generate function calling / tool usage training data",
            "Mode 5: Reasoning (CoT)": "Generate reasoning traces with <think> tags"
        }
        
        st.info(f"[INFO] {mode_descriptions[mode]}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_create_pipeline = False
                st.rerun()
        
        with col2:
            if st.button("Create Pipeline", type="primary", use_container_width=True):
                if pipeline_name:
                    # Extract mode number
                    mode_num = int(mode.split(":")[0].replace("Mode ", ""))
                    
                    import uuid
                    pipeline_id = str(uuid.uuid4())[:8]
                    pipeline = Pipeline(pipeline_name, pipeline_id)
                    pipeline.mode = mode_num
                    pipeline.mode_name = mode
                    
                    st.session_state.pipelines.append(pipeline)
                    st.session_state.active_pipeline = pipeline
                    st.session_state.show_create_pipeline = False
                    st.session_state.view = 'data_generation'
                    st.rerun()
                else:
                    st.error("Pipeline name required!")
        
        st.markdown("---")


# ==================== VIEW: DATA GENERATION ====================

def render_data_generation():
    """Render data generation view with chat interface and logs"""
    pipeline = st.session_state.active_pipeline
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em; margin: 0.5rem 0;">
                {pipeline.name}
            </h1>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span class="status-badge status-{pipeline.status}">{pipeline.status.upper()}</span>
                <span style="color: #666; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                    ID: {pipeline.id}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("[< DASHBOARD]", use_container_width=True):
            st.session_state.view = 'dashboard'
            st.rerun()
    
    # Main content: Chat interface + logs side by side
    col_chat, col_logs = st.columns([1, 1])
    
    with col_chat:
        render_chat_interface(pipeline)
    
    with col_logs:
        render_processing_logs(pipeline)


def render_chat_interface(pipeline: Pipeline):
    """Render chat-style interface for data generation"""
    st.markdown(f"""
    <div class="industrial-plate">
        <h3 style="color: #d6c0b6; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
            FORGE OPERATOR
        </h3>
        <div style="color: #8c6b5d; font-size: 0.7rem; margin-bottom: 1rem;">
            {pipeline.mode_name}
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection - Direct input for any Backboard model
    st.markdown("#### 🤖 MODEL SELECTION")
    
    st.markdown("""
    <div style="background: #111; border: 1px solid #333; padding: 0.75rem; margin-bottom: 1rem; font-size: 0.8rem;">
        <div style="color: #f96124; font-weight: bold; margin-bottom: 0.5rem;">💡 Direct Model Input</div>
        <div style="color: #888;">
            Enter any model from <a href="https://app.backboard.io/dashboard/model-library" target="_blank" style="color: #f96124;">Backboard Model Library</a><br>
            Format: <code style="color: #22c55e;">provider/model-name</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Direct model input
    model_input = st.text_input(
        "Model Name",
        value=pipeline.model if pipeline.model else "openai/gpt-4o",
        key="direct_model_input",
        placeholder="openai/gpt-4o"
    )
    
    # Validate and set model
    if model_input and '/' in model_input:
        if model_input != pipeline.model:
            pipeline.model = model_input
            add_log(pipeline, "INFO", f"Model set: {model_input}")
    else:
        st.warning("⚠️ Enter model as `provider/model-name`")
    
    # Show current selection
    if pipeline.model:
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #f96124; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
            <span style="color: #888; font-size: 0.7rem;">✓ ACTIVE MODEL:</span>
            <span style="color: #22c55e; font-weight: 700; margin-left: 0.5rem;">{pipeline.model}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Common models quick-select
    with st.expander("📋 Common Models (Quick Select)", expanded=False):
        common_models = [
            ("openai/gpt-4o", "GPT-4o - Fast & capable"),
            ("openai/gpt-4o-mini", "GPT-4o Mini - Fast & cheap"),
            ("anthropic/claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet - Best for coding"),
            ("anthropic/claude-3-5-haiku-20241022", "Claude 3.5 Haiku - Fast & cheap"),
            ("google/gemini-2.0-flash-exp", "Gemini 2.0 Flash - Free & fast"),
            ("deepseek/deepseek-chat", "DeepSeek Chat - Great value"),
            ("qwen/qwen-2.5-coder-32b-instruct", "Qwen 2.5 Coder 32B - Code specialist"),
            ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B - Open source"),
        ]
        
        cols = st.columns(2)
        for idx, (model_id, desc) in enumerate(common_models):
            with cols[idx % 2]:
                if st.button(f"📌 {model_id.split('/')[1][:20]}", key=f"quick_{idx}", use_container_width=True, help=desc):
                    pipeline.model = model_id
                    st.rerun()
    
    st.markdown("---")
    
    # Generation parameters
    st.markdown("#### Generation Parameters")
    
    # Mode-specific parameters
    if pipeline.mode == 1:
        # Mode 1: General Chat
        topic = st.text_input(
            "Topic/Domain",
            placeholder="e.g., Python programming, Customer support",
            help="What domain should the data cover?",
            key="gen_topic",
            value=pipeline.config.get("topic", "")
        )
        pipeline.config["topic"] = topic
        
        style = st.selectbox(
            "Generation Style",
            ["general_qa", "conversation", "instruction_following"],
            help="What type of conversational style?",
            key="gen_style"
        )
        pipeline.config["style"] = style
        
        # Custom instructions for Mode 1
        custom_instructions = st.text_area(
            "📝 Custom Instructions (Optional)",
            placeholder="Add specific instructions for the AI, e.g., 'Focus on beginner-friendly explanations' or 'Include code examples in responses'",
            help="Additional instructions to guide the generation",
            key="custom_instructions_1",
            value=pipeline.config.get("custom_instructions", "")
        )
        pipeline.config["custom_instructions"] = custom_instructions
        
    elif pipeline.mode == 2:
        # Mode 2: RAG / Knowledge Injection
        topic = st.text_input(
            "Topic/Domain",
            placeholder="e.g., Product documentation, Research papers",
            key="gen_topic",
            value=pipeline.config.get("topic", "")
        )
        pipeline.config["topic"] = topic
        
        st.markdown("#### [DOCUMENT_UPLOAD]")
        uploaded_files = st.file_uploader(
            "Upload documents for RAG",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx"],
            key="doc_upload"
        )
        
        if uploaded_files:
            st.success(f"[READY] {len(uploaded_files)} document(s) uploaded")
            pipeline.config["uploaded_files"] = uploaded_files
        
        style = st.selectbox(
            "Generation Style",
            ["qa_from_docs", "summarization"],
            key="gen_style"
        )
        pipeline.config["style"] = style
        
        # Custom instructions for Mode 2
        custom_instructions = st.text_area(
            "[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions, e.g., 'Extract technical details' or 'Focus on practical applications'",
            help="Additional instructions to guide document-grounded generation",
            key="custom_instructions_2",
            value=pipeline.config.get("custom_instructions", "")
        )
        pipeline.config["custom_instructions"] = custom_instructions
        
    elif pipeline.mode == 3:
        # Mode 3: Code Specialist
        topic = st.text_input(
            "Coding Topic",
            placeholder="e.g., Binary trees, API design, Database queries",
            key="gen_topic",
            value=pipeline.config.get("topic", "")
        )
        pipeline.config["topic"] = topic
        
        code_language = st.selectbox(
            "Programming Language",
            ["python", "javascript", "java", "cpp", "rust", "go", "typescript"],
            key="code_lang"
        )
        pipeline.config["code_language"] = code_language
        
        style = st.selectbox(
            "Code Style",
            ["algorithm", "debugging", "implementation"],
            key="gen_style"
        )
        pipeline.config["style"] = style
        
        # Custom instructions for Mode 3
        custom_instructions = st.text_area(
            "[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions, e.g., 'Include comprehensive docstrings' or 'Focus on error handling'",
            help="Additional instructions to guide code generation",
            key="custom_instructions_3",
            value=pipeline.config.get("custom_instructions", "")
        )
        pipeline.config["custom_instructions"] = custom_instructions
        
        # Suggest code models
        if not pipeline.model or "qwen" not in pipeline.model.lower():
            st.info("[TIP] Select a Qwen code model for best results (e.g., qwen-2.5-coder-32b)")
        
    elif pipeline.mode == 4:
        # Mode 4: Agent / Tool Use
        topic = st.text_input(
            "Tool Use Scenario",
            placeholder="e.g., Web search, Calculator, File operations",
            key="gen_topic",
            value=pipeline.config.get("topic", "")
        )
        pipeline.config["topic"] = topic
        
        st.markdown("#### [TOOL_DEFINITIONS]")
        
        # Example tools
        with st.expander("[EXAMPLE_SCHEMAS]"):
            st.code('''
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for information",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"}
      },
      "required": ["query"]
    }
  }
}
            ''', language="json")
        
        tools_json = st.text_area(
            "Tool Definitions (JSON array)",
            height=150,
            placeholder='[{"type": "function", "function": {...}}]',
            key="tools_json"
        )
        
        if tools_json:
            try:
                import json
                pipeline.config["tools"] = json.loads(tools_json)
                st.success(f"[READY] {len(pipeline.config['tools'])} tool(s) configured")
            except json.JSONDecodeError as e:
                st.error(f"[ERROR] Invalid JSON: {e}")
        
        style = st.selectbox(
            "Agent Style",
            ["tool_selection", "multi_step"],
            key="gen_style"
        )
        pipeline.config["style"] = style
        
        # Custom instructions for Mode 4
        custom_instructions = st.text_area(
            "[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions, e.g., 'Prefer sequential tool calls' or 'Include error handling scenarios'",
            help="Additional instructions to guide agent/tool generation",
            key="custom_instructions_4",
            value=pipeline.config.get("custom_instructions", "")
        )
        pipeline.config["custom_instructions"] = custom_instructions
        
    elif pipeline.mode == 5:
        # Mode 5: Reasoning
        topic = st.text_input(
            "Reasoning Topic",
            placeholder="e.g., Logic puzzles, Math problems, Analysis tasks",
            key="gen_topic",
            value=pipeline.config.get("topic", "")
        )
        pipeline.config["topic"] = topic
        
        style = st.selectbox(
            "Reasoning Style",
            ["logical_reasoning", "math_reasoning", "analysis"],
            key="gen_style"
        )
        pipeline.config["style"] = style
        
        # Custom instructions for Mode 5
        custom_instructions = st.text_area(
            "[CUSTOM_INSTRUCTIONS]",
            placeholder="Add specific instructions, e.g., 'Show detailed step-by-step reasoning' or 'Include edge case analysis'",
            help="Additional instructions to guide reasoning generation",
            key="custom_instructions_5",
            value=pipeline.config.get("custom_instructions", "")
        )
        pipeline.config["custom_instructions"] = custom_instructions
        
        # Suggest reasoning models
        if not pipeline.model or ("deepseek" not in pipeline.model.lower() and "o1" not in pipeline.model.lower()):
            st.info("[TIP] Select DeepSeek R1 or OpenAI O1 for reasoning with <think> tags")
    
    # ==========================================
    # RAG Add-on (Available for all modes except Mode 2 which has it built-in)
    # ==========================================
    if pipeline.mode != 2:
        st.markdown("---")
        st.markdown("#### KNOWLEDGE INJECTION (RAG)")
        
        pipeline.use_rag = st.checkbox(
            "Enable RAG / Knowledge Injection",
            value=pipeline.use_rag,
            help="Ground generation in uploaded documents for more accurate, factual outputs",
            key="enable_rag"
        )
        
        if pipeline.use_rag:
            st.markdown("""
            <div style="background: #1a1008; border-left: 3px solid #ff6b35; padding: 0.5rem; margin-bottom: 1rem; font-size: 0.85rem; color: #d6c0b6;">
                [DOC] Upload documents to ground the generated data in real content. 
                The AI will use this as context when generating samples.
            </div>
            """, unsafe_allow_html=True)
            
            rag_files = st.file_uploader(
                "Upload Reference Documents",
                accept_multiple_files=True,
                type=["pdf", "txt", "md", "docx"],
                key="rag_addon_docs"
            )
            
            if rag_files:
                st.success(f"[READY] {len(rag_files)} document(s) ready for knowledge injection")
                pipeline.config["uploaded_files"] = rag_files
                
                # Preview uploaded docs
                with st.expander("[UPLOADED_DOCS]"):
                    for f in rag_files:
                        st.write(f"• {f.name} ({f.size / 1024:.1f} KB)")
    
    # ==========================================
    # Web Search (Available for ALL modes)
    # ==========================================
    st.markdown("---")
    st.markdown("#### 🔍 WEB SEARCH CONTEXT")
    
    enable_web_search = st.checkbox(
        "Enable Web Search",
        value=pipeline.config.get("enable_web_search", False),
        help="Search the web for relevant context during generation (uses Perplexity API)",
        key="enable_web_search"
    )
    pipeline.config["enable_web_search"] = enable_web_search
    
    if enable_web_search:
        st.markdown("""
        <div style="background: #0a1628; border-left: 3px solid #3b82f6; padding: 0.5rem; margin-bottom: 1rem; font-size: 0.85rem; color: #a0c4ff;">
            🌐 Web search will be performed when generation starts. Results will be injected as context for the AI.
        </div>
        """, unsafe_allow_html=True)
        
        web_search_query = st.text_area(
            "Search Query",
            placeholder="Enter your search query, e.g., 'Latest Python 3.12 features' or 'Best practices for REST API design'",
            help="What should we search for? This query will be sent to Perplexity to gather relevant context.",
            key="web_search_query",
            value=pipeline.config.get("web_search_query", pipeline.config.get("topic", "")),
            height=80
        )
        pipeline.config["web_search_query"] = web_search_query
        
        if not web_search_query.strip():
            st.warning("⚠️ Enter a search query to enable web search")
        else:
            st.success(f"✓ Will search: '{web_search_query[:50]}{'...' if len(web_search_query) > 50 else ''}'")
    
    # Common parameters
    st.markdown("---")
    num_samples = st.number_input(
        "Number of Samples",
        min_value=1,
        max_value=10000,
        value=100,
        step=10,
        key="gen_samples"
    )
    
    # Dataset append option
    st.markdown("#### DATASET_OPTIONS")
    
    # Check for existing datasets
    load_existing_datasets()
    dataset_options = ["Create New Dataset"] + [
        f"{ds.name} ({ds.sample_count} samples)" 
        for ds in st.session_state.datasets.values()
    ]
    dataset_ids = [None] + list(st.session_state.datasets.keys())
    
    selected_dataset_idx = st.selectbox(
        "Target Dataset",
        range(len(dataset_options)),
        format_func=lambda i: dataset_options[i],
        help="Append to existing dataset (with memory for deduplication) or create new",
        key="target_dataset"
    )
    
    if selected_dataset_idx > 0:
        pipeline.dataset_id = dataset_ids[selected_dataset_idx]
        dataset = st.session_state.datasets[pipeline.dataset_id]
        pipeline.thread_id = dataset.thread_id
        pipeline.assistant_id = dataset.assistant_id
        st.info(f"[NOTE] Will append to '{dataset.name}' (currently {dataset.sample_count} samples). Memory ensures no duplicates!")
    else:
        pipeline.dataset_id = None
    
    # Show current config for debugging
    with st.expander("[DEBUG_CONFIG]", expanded=False):
        st.write(f"**Model:** {pipeline.model or 'Not selected'}")
        st.write(f"**Topic:** '{pipeline.config.get('topic', '')}' (empty: {not pipeline.config.get('topic')})")
        st.write(f"**Mode:** {pipeline.mode} - {pipeline.mode_name}")
        st.write(f"**RAG Enabled:** {pipeline.use_rag}")
        st.write(f"**Dataset:** {pipeline.dataset_id or 'New'}")
        st.write(f"**Status:** {pipeline.status}")
    
    # Generate button
    st.markdown("---")
    
    # Pre-validation visual
    ready_to_generate = True
    issues = []
    if not pipeline.model:
        issues.append("[ERR] No model selected")
        ready_to_generate = False
    if not pipeline.config.get("topic"):
        issues.append("[ERR] No topic provided")
        ready_to_generate = False
    if pipeline.mode == 2:
        has_docs = pipeline.config.get("uploaded_files")
        has_web_search = pipeline.config.get("enable_web_search") and pipeline.config.get("web_search_query", "").strip()
        if not has_docs and not has_web_search:
            issues.append("[ERR] Upload documents or enable web search for RAG mode")
            ready_to_generate = False
    if pipeline.mode == 4 and not pipeline.config.get("tools"):
        issues.append("[ERR] No tools defined for Agent mode")
        ready_to_generate = False
    
    if issues:
        st.warning("**Before generating:**\n" + "\n".join(issues))
    else:
        st.success("[READY] Ready to generate!")
    
    if st.button("[INITIATE_GENERATION]", type="primary", use_container_width=True, key="start_gen_btn", disabled=not ready_to_generate):
        # Start generation with spinner
        with st.spinner("[PROCESSING] Generating data..."):
            pipeline.status = "running"
            pipeline.target_samples = num_samples
            add_log(pipeline, "INFO", f"Starting {pipeline.mode_name}")
            add_log(pipeline, "INFO", f"Target: {num_samples} samples on '{pipeline.config['topic']}'")
            add_log(pipeline, "INFO", f"Using model: {pipeline.model}")
            
            # Check if appending to existing dataset
            existing_thread_id = pipeline.thread_id if pipeline.dataset_id else None
            existing_assistant_id = pipeline.assistant_id if pipeline.dataset_id else None
            if existing_thread_id:
                thread_preview = str(existing_thread_id)[:8] if existing_thread_id else ""
                add_log(pipeline, "INFO", f"Appending to dataset with memory continuity (thread: {thread_preview}...)")
            
            # Actually generate the data
            try:
                from src.backboard_manager import BackboardManager
                from src.generator import DataGenerator
                
                add_log(pipeline, "INFO", "Initializing Backboard connection...")
                manager = BackboardManager(api_key=st.session_state.api_key)
                generator = DataGenerator(manager)
                
                add_log(pipeline, "INFO", "Starting sample generation...")
                
                # Parse model provider and name
                model_parts = pipeline.model.split("/")
                llm_provider = model_parts[0] if len(model_parts) > 1 else "openai"
                model_name = model_parts[1] if len(model_parts) > 1 else pipeline.model
                
                samples = []
                
                # Perform web search if enabled (for all modes)
                web_search_context = None
                if pipeline.config.get("enable_web_search") and pipeline.config.get("web_search_query"):
                    search_query = pipeline.config["web_search_query"]
                    add_log(pipeline, "INFO", f"Performing web search: '{search_query[:50]}...'")
                    try:
                        import asyncio
                        search_results = asyncio.run(manager.web_search(search_query))
                        web_search_context = search_results.get("content", "")
                        sources = search_results.get("sources", [])
                        if web_search_context:
                            add_log(pipeline, "INFO", f"Web search completed - {len(sources)} sources found")
                            # Store results for reference
                            pipeline.config["web_search_results"] = search_results
                        else:
                            add_log(pipeline, "WARN", "Web search returned no content")
                    except Exception as e:
                        add_log(pipeline, "ERROR", f"Web search failed: {str(e)}")
                
                # Process RAG documents for non-Mode-2 modes (Mode 2 handles this differently)
                document_context = None
                if pipeline.mode != 2 and pipeline.use_rag and pipeline.config.get("uploaded_files"):
                    import asyncio
                    uploaded_files = pipeline.config.get("uploaded_files", [])
                    add_log(pipeline, "INFO", f"Processing {len(uploaded_files)} document(s) for RAG...")
                    
                    doc_contents = []
                    for file in uploaded_files:
                        try:
                            # Upload document to extract content
                            doc_id = asyncio.run(manager.upload_document(
                                file_path=file.name,
                                file_content=file.read(),
                                file_name=file.name
                            ))
                            # Get the extracted content
                            if hasattr(manager, '_document_cache') and doc_id in manager._document_cache:
                                doc_data = manager._document_cache[doc_id]
                                doc_name = doc_data.get('name', file.name)
                                doc_content = doc_data.get('content', '')
                                if doc_content:
                                    doc_contents.append(f"=== DOCUMENT: {doc_name} ===\n{doc_content}")
                                    add_log(pipeline, "SUCCESS", f"Extracted content from: {file.name}")
                        except Exception as e:
                            add_log(pipeline, "ERROR", f"Failed to process {file.name}: {str(e)}")
                    
                    if doc_contents:
                        document_context = "\n\n".join(doc_contents)
                        add_log(pipeline, "INFO", f"Document context ready: {len(document_context)} characters")
                
                # Combine web search and document context
                combined_context = None
                if web_search_context and document_context:
                    combined_context = f"{document_context}\n\n{web_search_context}"
                    add_log(pipeline, "INFO", "Using combined context: documents + web search")
                elif document_context:
                    combined_context = document_context
                    add_log(pipeline, "INFO", "Using document context only")
                elif web_search_context:
                    combined_context = web_search_context
                    # Already logged above
                
                # Use combined_context for web_search_context parameter
                if combined_context:
                    web_search_context = combined_context
                
                # Route to appropriate generation method
                if pipeline.mode == 1:
                    add_log(pipeline, "INFO", f"Mode 1: Generating {num_samples} chat samples...")
                    custom_instructions = pipeline.config.get("custom_instructions", "")
                    if custom_instructions:
                        add_log(pipeline, "INFO", f"Using custom instructions: {custom_instructions[:50]}...")
                    samples = generator.generate_mode1_samples(
                        num_samples=num_samples,
                        topic=pipeline.config["topic"],
                        style=pipeline.config.get("style", "general_qa"),
                        llm_provider=llm_provider,
                        model_name=model_name,
                        custom_prompt=custom_instructions if custom_instructions else None,
                        existing_thread_id=existing_thread_id,
                        existing_assistant_id=existing_assistant_id,
                        web_search_context=web_search_context
                    )
                    
                elif pipeline.mode == 2:
                    add_log(pipeline, "INFO", "Mode 2: Processing context sources...")
                    document_ids = []
                    import asyncio
                    
                    # Upload documents if provided
                    for file in pipeline.config.get("uploaded_files", []):
                        doc_id = asyncio.run(manager.upload_document(
                            file_path=file.name,
                            file_content=file.read(),
                            file_name=file.name
                        ))
                        document_ids.append(doc_id)
                        add_log(pipeline, "SUCCESS", f"Uploaded document: {file.name}")
                    
                    # Check what context sources we have
                    has_web_search = bool(web_search_context)
                    
                    if document_ids and has_web_search:
                        add_log(pipeline, "INFO", f"Using {len(document_ids)} document(s) + web search context")
                    elif document_ids:
                        add_log(pipeline, "INFO", f"Using {len(document_ids)} document(s)")
                    elif has_web_search:
                        add_log(pipeline, "INFO", "Using web search context only")
                    else:
                        add_log(pipeline, "WARNING", "No context sources provided")
                    
                    pipeline.config["document_ids"] = document_ids
                    
                    source_count = len(document_ids) + (1 if has_web_search else 0)
                    add_log(pipeline, "INFO", f"Generating {num_samples} RAG samples from {source_count} source(s)...")
                    custom_instructions = pipeline.config.get("custom_instructions", "")
                    if custom_instructions:
                        add_log(pipeline, "INFO", f"Using custom instructions: {custom_instructions[:50]}...")
                    samples = generator.generate_mode2_samples(
                        num_samples=num_samples,
                        document_ids=document_ids,
                        topic=pipeline.config["topic"],
                        style=pipeline.config.get("style", "qa_from_docs"),
                        llm_provider=llm_provider,
                        model_name=model_name,
                        custom_prompt=custom_instructions if custom_instructions else None,
                        existing_thread_id=existing_thread_id,
                        existing_assistant_id=existing_assistant_id,
                        web_search_context=web_search_context
                    )
                    
                elif pipeline.mode == 3:
                    add_log(pipeline, "INFO", f"Mode 3: Generating {num_samples} code samples...")
                    custom_instructions = pipeline.config.get("custom_instructions", "")
                    if custom_instructions:
                        add_log(pipeline, "INFO", f"Using custom instructions: {custom_instructions[:50]}...")
                    samples = generator.generate_mode3_samples(
                        num_samples=num_samples,
                        topic=pipeline.config["topic"],
                        style=pipeline.config.get("style", "algorithm"),
                        code_language=pipeline.config.get("code_language", "python"),
                        llm_provider=llm_provider or "qwen",
                        model_name=model_name or "qwen-2.5-coder-32b-instruct",
                        custom_prompt=custom_instructions if custom_instructions else None,
                        existing_thread_id=existing_thread_id,
                        existing_assistant_id=existing_assistant_id,
                        web_search_context=web_search_context
                    )
                    
                elif pipeline.mode == 4:
                    add_log(pipeline, "INFO", f"Mode 4: Generating {num_samples} agent samples...")
                    custom_instructions = pipeline.config.get("custom_instructions", "")
                    if custom_instructions:
                        add_log(pipeline, "INFO", f"Using custom instructions: {custom_instructions[:50]}...")
                    samples = generator.generate_mode4_samples(
                        num_samples=num_samples,
                        tools=pipeline.config.get("tools", []),
                        topic=pipeline.config["topic"],
                        style=pipeline.config.get("style", "tool_selection"),
                        llm_provider=llm_provider,
                        model_name=model_name,
                        custom_prompt=custom_instructions if custom_instructions else None,
                        existing_thread_id=existing_thread_id,
                        existing_assistant_id=existing_assistant_id,
                        web_search_context=web_search_context
                    )
                    
                elif pipeline.mode == 5:
                    add_log(pipeline, "INFO", f"Mode 5: Generating {num_samples} reasoning samples...")
                    custom_instructions = pipeline.config.get("custom_instructions", "")
                    if custom_instructions:
                        add_log(pipeline, "INFO", f"Using custom instructions: {custom_instructions[:50]}...")
                    samples = generator.generate_mode5_samples(
                        num_samples=num_samples,
                        topic=pipeline.config["topic"],
                        style=pipeline.config.get("style", "logical_reasoning"),
                        llm_provider=llm_provider or "deepseek",
                        model_name=model_name or "deepseek-r1",
                        custom_prompt=custom_instructions if custom_instructions else None,
                        existing_thread_id=existing_thread_id,
                        existing_assistant_id=existing_assistant_id,
                        web_search_context=web_search_context
                    )
                
                # Update pipeline with results
                pipeline.samples_generated = len(samples)
                pipeline.status = "complete"
                add_log(pipeline, "SUCCESS", f"Generation complete! Generated {len(samples)} samples")
                
                # Get the thread/assistant IDs from the generator for future appends
                new_thread_id = generator.last_thread_id
                new_assistant_id = generator.last_assistant_id
                
                # Handle dataset creation/update
                output_dir = Path("data")
                output_dir.mkdir(exist_ok=True)
                
                if pipeline.dataset_id:
                    # Appending to existing dataset
                    dataset = st.session_state.datasets[pipeline.dataset_id]
                    dataset.add_samples(samples)
                    dataset.save()
                    add_log(pipeline, "SUCCESS", f"Appended {len(samples)} samples to dataset '{dataset.name}' (total: {dataset.sample_count})")
                    output_file = dataset.file_path
                else:
                    # Create new dataset
                    output_file = output_dir / f"{pipeline.id}_{pipeline.name.replace(' ', '_')}.jsonl"
                    generator.export_to_jsonl(output_path=str(output_file), format_type="alpaca")
                    
                    # Create dataset object for future appends
                    new_dataset = Dataset(
                        id=pipeline.id,
                        name=pipeline.name,
                        mode=pipeline.mode,
                        mode_name=pipeline.mode_name,
                        file_path=str(output_file),
                        samples=[s.to_alpaca() for s in samples],
                        created_at=datetime.now(),
                        thread_id=new_thread_id,
                        assistant_id=new_assistant_id
                    )
                    st.session_state.datasets[new_dataset.id] = new_dataset
                    new_dataset.save()  # Ensure metadata is written to disk immediately
                    add_log(pipeline, "SUCCESS", f"Created new dataset: {new_dataset.name}")
                
                add_log(pipeline, "SUCCESS", f"Exported to: {output_file}")
                
                st.success(f"[SUCCESS] Generated {len(samples)} samples!")
                st.rerun()
                
            except Exception as e:
                pipeline.status = "error"
                add_log(pipeline, "ERROR", f"Generation failed: {str(e)}")
                st.error(f"[ERROR] {str(e)}")
                import traceback
                add_log(pipeline, "ERROR", traceback.format_exc())
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_processing_logs(pipeline: Pipeline):
    """Render processing logs terminal"""
    st.markdown("""
    <div class="industrial-plate">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="color: #d6c0b6; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">
                SYSTEM LOGS
            </h3>
            <span class="status-badge status-running" style="font-size: 0.65rem;">LIVE</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Render logs
    log_html = '<div class="terminal-log">'
    
    if not pipeline.logs:
        log_html += '<div style="color: #5c3a2e; font-style: italic;">Waiting for operations...</div>'
    else:
        for log in pipeline.logs[-20:]:  # Show last 20 logs
            level_class = f"log-level-{log['level'].lower()}"
            log_html += f"""
            <div class="log-entry">
                <span class="log-time">{log['time']}</span>
                <span class="{level_class}">{log['level']}</span>
                <span>{log['message']}</span>
            </div>
            """
    
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)



# ==================== MAIN APP ====================

def main():
    st.set_page_config(
        page_title="ModelForge",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load theme
    load_forge_theme()
    
    # Initialize state
    init_session_state()
    
    # Route to appropriate view
    if st.session_state.view == 'setup':
        render_setup_view()
    elif st.session_state.view == 'dashboard':
        render_dashboard()
    elif st.session_state.view == 'data_generation':
        render_data_generation()
    elif st.session_state.view == 'data_viewer':
        render_data_viewer()
    elif st.session_state.view == 'fine_tuning':
        render_finetuning()
    elif st.session_state.view == 'inference':
        render_inference()
    elif st.session_state.view == 'hf_deploy':
        render_hf_deploy()


# ==================== MEMORY VISUALIZATION ====================

def render_memory_visualization(dataset):
    """
    Render interactive node-based memory visualization for a dataset showing
    how Backboard's memory has been used to prevent data duplication.
    """
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #f96124;">MEMORY_NETWORK</h3>
        <span style="color: #8c6b5d; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
            [BACKBOARD.IO MEMORY SYSTEM]
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Check if dataset has assistant_id for memory lookup
    if not dataset.assistant_id:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #666;">
            <div style="color: #888; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                <span style="color: #eab308;">[WARNING]</span> No assistant ID linked to this dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Try to get API key
    api_key = st.session_state.get('api_key') or os.getenv("BACKBOARD_API_KEY")
    if not api_key:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #ef4444;">
            <div style="color: #888; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                <span style="color: #ef4444;">[ERROR]</span> Backboard API key not configured.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Fetch memories from Backboard API
    with st.spinner("Fetching memories from Backboard..."):
        try:
            manager = BackboardManager(api_key=api_key)
            memories_data = run_async(manager.get_all_memories(dataset.assistant_id))
        except Exception as e:
            st.error(f"[ERROR] Failed to fetch memories: {e}")
            return

    memories = memories_data.get("memories", [])
    total_count = memories_data.get("total_count", len(memories))

    # Memory Stats Header
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #f96124;">{total_count}</div>
            <div class="stat-label">MEMORIES STORED</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stat2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #22c55e;">{dataset.sample_count}</div>
            <div class="stat-label">SAMPLES GENERATED</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stat3:
        efficiency = (total_count / dataset.sample_count * 100) if dataset.sample_count > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #3b82f6;">{efficiency:.1f}%</div>
            <div class="stat-label">MEMORY DENSITY</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stat4:
        thread_id_str = str(dataset.thread_id) if dataset.thread_id else ""
        thread_display = thread_id_str[:8] if thread_id_str else "N/A"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #a855f7; font-size: 1.2rem;">{thread_display}...</div>
            <div class="stat-label">THREAD ID</div>
        </div>
        """, unsafe_allow_html=True)

    # Node Graph Visualization
    if memories:
        # Prepare memory data for JavaScript (limit to 50 for performance)
        display_memories = memories[:50]
        nodes_data = []
        for i, memory in enumerate(display_memories):
            content = str(memory.get("content", ""))
            memory_id = str(memory.get("id", "unknown"))
            score = memory.get("score", 0) or 0
            created_at = str(memory.get("created_at", ""))[:19]

            # Escape content for JavaScript
            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

            nodes_data.append({
                "id": i,
                "memoryId": memory_id[:12],
                "content": escaped_content[:300],
                "score": float(score) if score else 0,
                "createdAt": created_at
            })

        nodes_json = json.dumps(nodes_data)

        # Create the interactive node visualization
        graph_html = f'''
        <div id="memory-graph-container" style="position: relative; width: 100%; height: 500px; background: #0a0a0a; border: 1px solid #333; overflow: hidden;">
            <canvas id="memoryCanvas" style="width: 100%; height: 100%;"></canvas>
            <div id="tooltip" style="
                display: none;
                position: absolute;
                background: #141414;
                border: 1px solid #f96124;
                border-left: 4px solid #f96124;
                padding: 12px;
                max-width: 350px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                color: #ccc;
                z-index: 1000;
                box-shadow: 0 4px 20px rgba(249, 97, 36, 0.3);
                pointer-events: none;
            "></div>
            <div style="position: absolute; bottom: 10px; left: 10px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666;">
                [HOVER NODES TO VIEW MEMORY CONTENT]
            </div>
            <div style="position: absolute; top: 10px; right: 10px; font-family: 'JetBrains Mono', monospace; font-size: 10px;">
                <span style="color: #22c55e;">● HIGH</span>
                <span style="color: #eab308; margin-left: 10px;">● MED</span>
                <span style="color: #f96124; margin-left: 10px;">● LOW</span>
            </div>
        </div>

        <script>
        (function() {{
            const nodes = {nodes_json};
            const canvas = document.getElementById('memoryCanvas');
            const container = document.getElementById('memory-graph-container');
            const tooltip = document.getElementById('tooltip');
            const ctx = canvas.getContext('2d');

            // Set canvas size
            function resizeCanvas() {{
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
            }}
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);

            // Node properties
            const nodeRadius = 8;
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;

            // Position nodes in a force-directed-like layout
            nodes.forEach((node, i) => {{
                const angle = (i / nodes.length) * Math.PI * 2;
                const radius = 120 + Math.random() * 100;
                node.x = centerX + Math.cos(angle) * radius + (Math.random() - 0.5) * 60;
                node.y = centerY + Math.sin(angle) * radius + (Math.random() - 0.5) * 60;
                node.vx = 0;
                node.vy = 0;
            }});

            // Create connections (connect nearby nodes and sequential nodes)
            const connections = [];
            for (let i = 0; i < nodes.length; i++) {{
                // Connect to next node (temporal connection)
                if (i < nodes.length - 1) {{
                    connections.push([i, i + 1]);
                }}
                // Connect to some random nearby nodes
                for (let j = i + 2; j < Math.min(i + 4, nodes.length); j++) {{
                    if (Math.random() > 0.5) {{
                        connections.push([i, j]);
                    }}
                }}
            }}

            // Get color based on score
            function getNodeColor(score) {{
                if (score >= 0.8) return '#22c55e';
                if (score >= 0.5) return '#eab308';
                return '#f96124';
            }}

            // Animation loop
            let hoveredNode = null;
            let animationFrame = 0;

            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                animationFrame++;

                // Draw grid pattern
                ctx.strokeStyle = '#1a1a1a';
                ctx.lineWidth = 1;
                for (let x = 0; x < canvas.width; x += 30) {{
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, canvas.height);
                    ctx.stroke();
                }}
                for (let y = 0; y < canvas.height; y += 30) {{
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(canvas.width, y);
                    ctx.stroke();
                }}

                // Draw connections
                connections.forEach(([i, j]) => {{
                    const nodeA = nodes[i];
                    const nodeB = nodes[j];
                    const gradient = ctx.createLinearGradient(nodeA.x, nodeA.y, nodeB.x, nodeB.y);
                    gradient.addColorStop(0, getNodeColor(nodeA.score) + '40');
                    gradient.addColorStop(1, getNodeColor(nodeB.score) + '40');

                    ctx.beginPath();
                    ctx.moveTo(nodeA.x, nodeA.y);
                    ctx.lineTo(nodeB.x, nodeB.y);
                    ctx.strokeStyle = gradient;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }});

                // Draw nodes
                nodes.forEach((node, i) => {{
                    const isHovered = hoveredNode === i;
                    const color = getNodeColor(node.score);
                    const radius = isHovered ? nodeRadius * 1.8 : nodeRadius;

                    // Glow effect
                    if (isHovered) {{
                        ctx.beginPath();
                        ctx.arc(node.x, node.y, radius + 15, 0, Math.PI * 2);
                        const glowGradient = ctx.createRadialGradient(node.x, node.y, radius, node.x, node.y, radius + 15);
                        glowGradient.addColorStop(0, color + '60');
                        glowGradient.addColorStop(1, 'transparent');
                        ctx.fillStyle = glowGradient;
                        ctx.fill();
                    }}

                    // Outer ring
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, radius + 2, 0, Math.PI * 2);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = isHovered ? 2 : 1;
                    ctx.stroke();

                    // Inner circle
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                    ctx.fillStyle = isHovered ? color : '#0a0a0a';
                    ctx.fill();

                    // Center dot
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, 3, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();

                    // Node number
                    ctx.fillStyle = isHovered ? '#000' : '#666';
                    ctx.font = '9px JetBrains Mono, monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(i + 1).padStart(2, '0'), node.x, node.y);
                }});

                // Pulse effect on connections
                const pulsePos = (animationFrame % 100) / 100;
                connections.forEach(([i, j]) => {{
                    const nodeA = nodes[i];
                    const nodeB = nodes[j];
                    const px = nodeA.x + (nodeB.x - nodeA.x) * pulsePos;
                    const py = nodeA.y + (nodeB.y - nodeA.y) * pulsePos;

                    ctx.beginPath();
                    ctx.arc(px, py, 2, 0, Math.PI * 2);
                    ctx.fillStyle = '#f9612440';
                    ctx.fill();
                }});

                requestAnimationFrame(draw);
            }}

            // Mouse interaction
            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const mouseX = (e.clientX - rect.left) * scaleX;
                const mouseY = (e.clientY - rect.top) * scaleY;

                hoveredNode = null;
                for (let i = 0; i < nodes.length; i++) {{
                    const node = nodes[i];
                    const dist = Math.sqrt((mouseX - node.x) ** 2 + (mouseY - node.y) ** 2);
                    if (dist < nodeRadius + 5) {{
                        hoveredNode = i;
                        break;
                    }}
                }}

                if (hoveredNode !== null) {{
                    const node = nodes[hoveredNode];
                    const color = getNodeColor(node.score);
                    tooltip.innerHTML = `
                        <div style="color: ${{color}}; margin-bottom: 8px; font-size: 12px;">
                            [NODE ${{String(hoveredNode + 1).padStart(3, '0')}}]
                        </div>
                        <div style="color: #666; margin-bottom: 6px; font-size: 10px;">
                            ID: ${{node.memoryId}}... | ${{node.createdAt || 'N/A'}}
                        </div>
                        <div style="color: #aaa; line-height: 1.5; font-size: 11px; max-height: 150px; overflow: hidden;">
                            ${{node.content}}
                        </div>
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #333;">
                            <span style="color: ${{color}};">SCORE: ${{node.score.toFixed(2)}}</span>
                        </div>
                    `;
                    tooltip.style.display = 'block';

                    // Position tooltip
                    const tooltipX = e.clientX - container.getBoundingClientRect().left + 15;
                    const tooltipY = e.clientY - container.getBoundingClientRect().top + 15;

                    // Keep tooltip in bounds
                    const maxX = container.clientWidth - tooltip.offsetWidth - 10;
                    const maxY = container.clientHeight - tooltip.offsetHeight - 10;

                    tooltip.style.left = Math.min(tooltipX, maxX) + 'px';
                    tooltip.style.top = Math.min(tooltipY, maxY) + 'px';

                    canvas.style.cursor = 'pointer';
                }} else {{
                    tooltip.style.display = 'none';
                    canvas.style.cursor = 'default';
                }}
            }});

            canvas.addEventListener('mouseleave', () => {{
                hoveredNode = null;
                tooltip.style.display = 'none';
            }});

            draw();
        }})();
        </script>
        '''

        st.components.v1.html(graph_html, height=550)

        # Show count indicator
        if total_count > 50:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; color: #666; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                Displaying 50 of {total_count} memories
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 4px solid #666; text-align: center; padding: 2rem;">
            <div style="color: #666; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                [NO MEMORIES FOUND]
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Close button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("[CLOSE_MEMORY_VIEW]", use_container_width=True):
        memory_key = f"show_memory_{st.session_state.active_dataset}"
        st.session_state[memory_key] = False
        st.rerun()


# ==================== VIEW: DATA VIEWER ====================

def render_data_viewer():
    """Render the data viewer for browsing and managing datasets"""
    # Load existing datasets
    load_existing_datasets()
    
    # Header with navigation
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("""
        <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
            DATA VIEWER
        </h1>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("[DASHBOARD]", use_container_width=True):
            st.session_state.view = 'dashboard'
            st.rerun()
    with col3:
        if st.button("[NEW_PIPELINE]", use_container_width=True):
            st.session_state.show_create_modal = True
    
    st.markdown("---")
    
    # Dataset list and viewer
    col_list, col_viewer = st.columns([1, 2])
    
    with col_list:
        st.markdown("### LOCAL_DATASETS")
        
        if not st.session_state.datasets:
            st.info("[INFO] No datasets found. Generate some data first!")
        else:
            for dataset_id, dataset in st.session_state.datasets.items():
                is_active = st.session_state.active_dataset == dataset_id
                
                # Dataset card
                st.markdown(f"""
                <div class="industrial-plate" style="margin-bottom: 0.5rem; {'border: 2px solid #ff6b35;' if is_active else ''}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: white;">{dataset.name}</strong>
                            <div style="color: #8c6b5d; font-size: 0.75rem;">
                                {dataset.sample_count} samples • Mode {dataset.mode}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_view, col_add = st.columns(2)
                with col_view:
                    if st.button("[VIEW]", key=f"view_{dataset_id}", use_container_width=True):
                        st.session_state.active_dataset = dataset_id
                        st.rerun()
                with col_add:
                    if st.button("[ADD]", key=f"add_{dataset_id}", use_container_width=True):
                        # Create pipeline linked to this dataset
                        create_pipeline(f"Add to {dataset.name}", dataset_id=dataset_id)
                        st.rerun()
    
    with col_viewer:
        if st.session_state.active_dataset and st.session_state.active_dataset in st.session_state.datasets:
            dataset = st.session_state.datasets[st.session_state.active_dataset]
            
            st.markdown(f"### DATASET: {dataset.name}")
            st.markdown(f"**{dataset.sample_count} samples** | Mode: {dataset.mode_name}")
            st.markdown(f"File: `{dataset.file_path}`")
            
            # Sample browser
            if dataset.samples:
                st.markdown("---")
                
                # Pagination
                samples_per_page = 10
                total_pages = (len(dataset.samples) + samples_per_page - 1) // samples_per_page
                
                if 'sample_page' not in st.session_state:
                    st.session_state.sample_page = 0
                
                col_prev, col_page, col_next = st.columns([1, 2, 1])
                with col_prev:
                    if st.button("[PREV]", disabled=st.session_state.sample_page == 0):
                        st.session_state.sample_page -= 1
                        st.rerun()
                with col_page:
                    st.markdown(f"<div style='text-align: center; color: #d6c0b6;'>Page {st.session_state.sample_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
                with col_next:
                    if st.button("[NEXT]", disabled=st.session_state.sample_page >= total_pages - 1):
                        st.session_state.sample_page += 1
                        st.rerun()
                
                # Display samples
                start_idx = st.session_state.sample_page * samples_per_page
                end_idx = min(start_idx + samples_per_page, len(dataset.samples))
                
                for i, sample in enumerate(dataset.samples[start_idx:end_idx], start=start_idx + 1):
                    with st.expander(f"Sample {i}: {sample.get('instruction', 'No instruction')[:60]}..."):
                        st.markdown("**Instruction:**")
                        st.text(sample.get('instruction', ''))
                        
                        if sample.get('input'):
                            st.markdown("**Input:**")
                            st.text(sample.get('input', ''))
                        
                        st.markdown("**Output:**")
                        st.text(sample.get('output', ''))
                
                # Export options
                st.markdown("---")
                st.markdown("### EXPORT_OPTIONS")

                col_dl, col_mem, col_del = st.columns(3)
                with col_dl:
                    # Download button
                    jsonl_content = "\n".join([json.dumps(s, ensure_ascii=False) for s in dataset.samples])
                    st.download_button(
                        "[DOWNLOAD_JSONL]",
                        data=jsonl_content,
                        file_name=f"{dataset.name.replace(' ', '_')}.jsonl",
                        mime="application/json",
                        use_container_width=True
                    )
                with col_mem:
                    # Memory visualization toggle
                    memory_key = f"show_memory_{st.session_state.active_dataset}"
                    if st.button("[VIEW_MEMORY]", use_container_width=True):
                        st.session_state[memory_key] = not st.session_state.get(memory_key, False)
                        st.rerun()
                with col_del:
                    if st.button("[DELETE_DATASET]", use_container_width=True, type="secondary"):
                        # Delete file and remove from state
                        try:
                            Path(dataset.file_path).unlink()
                            del st.session_state.datasets[st.session_state.active_dataset]
                            st.session_state.active_dataset = None
                            st.success("[SUCCESS] Dataset deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"[ERROR] Error deleting: {e}")

                # Memory visualization section
                memory_key = f"show_memory_{st.session_state.active_dataset}"
                if st.session_state.get(memory_key, False):
                    render_memory_visualization(dataset)
        else:
            st.info("[INFO] Select a dataset to view its contents")
    
    # Show create modal if needed
    if st.session_state.get('show_create_modal', False):
        render_create_pipeline_modal()


# ==================== VIEW: FINE-TUNING (DOCKER) ====================

def render_finetuning():
    """Render the fine-tuning view using Docker-based Unsloth"""
    from src.finetuning_manager import (
        DockerFineTuningManager, TrainingConfig, UNSLOTH_MODELS,
        get_training_status, start_training_async, reset_training_status
    )
    
    # Load existing datasets
    load_existing_datasets()
    
    # Header with navigation
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("""
        <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
            MODEL FINE-TUNING
        </h1>
        <p style="color: #8c6b5d; font-size: 0.8rem;">
            Train custom models using Unsloth via Docker
        </p>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("[DASHBOARD]", use_container_width=True):
            st.session_state.view = 'dashboard'
            st.rerun()
    with col3:
        if st.button("[VIEW_DATA]", use_container_width=True):
            st.session_state.view = 'data_viewer'
            st.rerun()
    
    st.markdown("---")
    
    # Initialize manager
    manager = DockerFineTuningManager()
    
    # Check Docker status
    docker_ok, docker_msg = manager.check_docker()
    image_ok, image_msg = manager.check_unsloth_image() if docker_ok else (False, "Docker not available")
    container_running = manager.get_running_container() is not None
    
    # Status Cards
    st.markdown("### DOCKER STATUS")
    
    col_docker, col_image, col_container = st.columns(3)
    
    with col_docker:
        if docker_ok:
            st.markdown("""
            <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                <span style="color: #22c55e;">[READY]</span> <strong>Docker Ready</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="industrial-plate" style="border-left: 3px solid #ef4444;">
                <span style="color: #ef4444;">[OFFLINE]</span> <strong>Docker Not Found</strong>
                <div style="color: #8c6b5d; font-size: 0.75rem;">{docker_msg}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_image:
        if image_ok:
            st.markdown("""
            <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                <span style="color: #22c55e;">[READY]</span> <strong>Unsloth Image</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="industrial-plate" style="border-left: 3px solid #eab308;">
                <span style="color: #eab308;">[MISSING]</span> <strong>Image Not Found</strong>
                <div style="color: #8c6b5d; font-size: 0.75rem;">Pull required (~7GB)</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_container:
        if container_running:
            st.markdown("""
            <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                <span style="color: #22c55e;">[RUNNING]</span> <strong>Container Running</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="industrial-plate" style="border-left: 3px solid #8c6b5d;">
                <span style="color: #8c6b5d;">[STOPPED]</span> <strong>Container Stopped</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Docker controls
    if docker_ok:
        col_pull, col_start, col_stop = st.columns(3)
        
        with col_pull:
            if st.button("[PULL_IMAGE]", use_container_width=True, disabled=image_ok):
                with st.spinner("Pulling unsloth/unsloth image... This may take 5-10 minutes."):
                    success, msg = manager.pull_image()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                st.rerun()
        
        with col_start:
            if st.button("[START_CONTAINER]", use_container_width=True, disabled=not image_ok or container_running):
                with st.spinner("Starting container..."):
                    success, msg = manager.start_container()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                st.rerun()
        
        with col_stop:
            if st.button("[STOP_CONTAINER]", use_container_width=True, disabled=not container_running):
                with st.spinner("Stopping container..."):
                    manager.stop_container()
                    st.success("Container stopped")
                st.rerun()
    
    st.markdown("---")
    
    # Main content - two columns
    col_config, col_output = st.columns([1, 1])
    
    with col_config:
        st.markdown("### TRAINING_CONFIGURATION")
        
        # Model Selection
        st.markdown("#### BASE_MODEL")
        selected_model = st.selectbox(
            "Select Base Model",
            options=UNSLOTH_MODELS,
            index=0,
            help="Choose a base model to fine-tune. Smaller models (1B-3B) train faster."
        )
        
        st.markdown("---")
        
        # Dataset Selection
        st.markdown("#### TRAINING_DATASET")
        
        # Get available datasets
        available_datasets = []
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("*.jsonl"):
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        sample_count = sum(1 for _ in file)
                        available_datasets.append({
                            "name": f.stem,
                            "path": str(f),
                            "samples": sample_count
                        })
                except:
                    pass
        
        if not available_datasets:
            st.warning("No datasets found in data/ folder. Generate some training data first!")
            selected_dataset = None
        else:
            dataset_options = [f"{d['name']} ({d['samples']} samples)" for d in available_datasets]
            selected_idx = st.selectbox(
                "Select Dataset",
                options=range(len(dataset_options)),
                format_func=lambda x: dataset_options[x],
                help="Choose a JSONL dataset for training"
            )
            selected_dataset = available_datasets[selected_idx] if dataset_options else None
        
        st.markdown("---")
        
        # Training Parameters
        st.markdown("#### TRAINING_PARAMETERS")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            num_epochs = st.slider("Epochs", 1, 10, 3, help="More epochs = better learning, but slower")
            batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1)
            max_seq_length = st.selectbox("Max Seq Length", [512, 1024, 2048, 4096], index=1)
        
        with col_p2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                value=2e-4,
                format_func=lambda x: f"{x:.0e}"
            )
            lora_r = st.selectbox("LoRA Rank (r)", [8, 16, 32, 64], index=2, help="Higher = more capacity")
            lora_alpha = st.selectbox("LoRA Alpha", [16, 32, 64, 128], index=1, help="Usually 2x LoRA rank")
        
        st.markdown("---")
        
        # QLoRA Toggle
        st.markdown("#### QUANTIZATION")
        use_qlora = st.checkbox(
            "Use QLoRA (4-bit quantization)",
            value=True,
            help="QLoRA uses less VRAM but LoRA (16-bit) may give slightly better results. Recommended: QLoRA for 8GB GPUs."
        )
        
        st.markdown("---")
        
        # Custom Model Name
        st.markdown("#### OUTPUT_MODEL_NAME")
        custom_model_name = st.text_input(
            "Model Name (optional)",
            value="",
            placeholder="e.g., my-custom-model",
            help="Leave blank for auto-generated timestamp name"
        )
        
        st.markdown("---")
        
        # Training Actions
        st.markdown("#### TRAINING_ACTIONS")
        
        can_train = docker_ok and image_ok and container_running and selected_dataset is not None
        training_status = get_training_status()
        is_training = bool(training_status and training_status.is_running)
        
        # Create config with all new options
        config = TrainingConfig(
            model_name=selected_model,
            max_seq_length=max_seq_length,
            load_in_4bit=use_qlora,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            custom_model_name=custom_model_name,
        )
        
        col_train, col_notebook = st.columns(2)
        
        with col_train:
            if st.button(
                "[START_TRAINING]" if not is_training else "[TRAINING...]",
                type="primary",
                use_container_width=True,
                disabled=(not can_train) or is_training
            ):
                manager_with_config = DockerFineTuningManager(config)
                start_training_async(manager_with_config, selected_dataset['path'])
                st.rerun()
        
        with col_notebook:
            if st.button(
                "[GENERATE_NOTEBOOK]",
                use_container_width=True,
                disabled=(not can_train) or is_training
            ):
                manager_with_config = DockerFineTuningManager(config)
                notebook_path = manager_with_config.generate_training_notebook(selected_dataset['path'])
                st.session_state.generated_notebook = notebook_path
                st.success(f"Notebook: {Path(notebook_path).name}")
                st.rerun()
        
        if not can_train and not is_training:
            if not docker_ok:
                st.caption("[WARN] Start Docker Desktop first")
            elif not image_ok:
                st.caption("[WARN] Pull the Unsloth image first")
            elif not container_running:
                st.caption("[WARN] Start the container first")
            elif not selected_dataset:
                st.caption("[WARN] Select a dataset")
    
    with col_output:
        st.markdown("### TRAINING_STATUS")
        
        # Get training status
        training_status = get_training_status()
        
        if training_status and (training_status.is_running or training_status.is_complete or training_status.has_error):
            # Show training progress
            if training_status.is_running:
                st.markdown(f"""
                <div class="industrial-plate" style="border-left: 3px solid #f96124;">
                    <div style="color: #f96124; font-weight: bold;">[PROCESSING] Training in Progress</div>
                    <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                        <strong>Elapsed:</strong> {training_status.elapsed_time()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # GPU Stats Panel (only show if we have data)
                if training_status.gpu_memory_total > 0:
                    mem_pct = (training_status.gpu_memory_used / training_status.gpu_memory_total) * 100 if training_status.gpu_memory_total > 0 else 0
                    st.markdown(f"""
                    <div class="industrial-plate" style="border-left: 3px solid #3b82f6; margin-top: 0.5rem;">
                        <div style="color: #3b82f6; font-weight: bold; font-size: 0.9rem;">🖥️ GPU: {training_status.gpu_name or 'Unknown'}</div>
                        <div style="display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.8rem; color: #8c6b5d;">
                            <span>📊 VRAM: {training_status.gpu_memory_used:.0f}/{training_status.gpu_memory_total:.0f} MB ({mem_pct:.1f}%)</span>
                        </div>
                        <div style="display: flex; gap: 1rem; margin-top: 0.25rem; font-size: 0.8rem; color: #8c6b5d;">
                            <span>⚡ Util: {training_status.gpu_utilization}%</span>
                            <span>🌡️ Temp: {training_status.gpu_temperature}°C</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
            elif training_status.is_complete:
                st.markdown(f"""
                <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                    <div style="color: #22c55e; font-weight: bold;">[SUCCESS] Training Complete!</div>
                    <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                        <strong>Duration:</strong> {training_status.elapsed_time()}<br>
                        <strong>Output:</strong> {training_status.model_output_dir}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            elif training_status.has_error:
                st.markdown(f"""
                <div class="industrial-plate" style="border-left: 3px solid #ef4444;">
                    <div style="color: #ef4444; font-weight: bold;">[FAILED] Training Failed</div>
                    <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                        {training_status.error_message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Output log with scrollable container
            st.markdown("#### TRAINING_LOG")
            log_output = "\n".join(training_status.output_lines[-80:])  # Last 80 lines
            st.code(log_output, language="text", line_numbers=False)
            
            # Auto-refresh if running (moved here to ensure logs render first)
            if training_status.is_running:
                time.sleep(2)
                st.rerun()
            
            # Reset button (only when not running)
            if not training_status.is_running:
                if st.button("[NEW_TRAINING_RUN]", use_container_width=True):
                    reset_training_status()
                    st.rerun()
        
        elif container_running:
            st.markdown(f"""
            <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                <div style="color: #22c55e; font-weight: bold;">[CONTAINER_READY]</div>
                <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                    <strong>Jupyter:</strong> <a href="http://localhost:8888" target="_blank" style="color: #f96124;">http://localhost:8888</a><br>
                    <strong>Password:</strong> <code>modelforge</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("[OPEN_JUPYTER_LAB]", use_container_width=True):
                import webbrowser
                webbrowser.open("http://localhost:8888")
            
            st.markdown("---")
            
            # Show generated notebook info
            if 'generated_notebook' in st.session_state and st.session_state.generated_notebook:
                notebook_name = Path(st.session_state.generated_notebook).name
                st.markdown(f"""
                <div class="industrial-plate" style="border-left: 3px solid #f96124;">
                    <div style="color: white; font-weight: bold;">[LATEST_NOTEBOOK]</div>
                    <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                        <code>{notebook_name}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### [INFO] Two Ways to Train")
            st.markdown("""
            **Option 1: Direct Training** (Recommended)
            - Click "[START_TRAINING]" 
            - Watch progress in real-time
            - Model auto-saves to `work/models/`
            
            **Option 2: Jupyter Notebook**
            - Click "[GENERATE_NOTEBOOK]"
            - Open Jupyter Lab
            - Run cells manually for more control
            """)
        else:
            st.markdown("""
            <div class="industrial-plate" style="opacity: 0.7;">
                <div style="color: #8c6b5d; text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐳</div>
                    <div>Start the container to begin training</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick guide
        st.markdown("#### [QUICK_START]")
        st.markdown("""
        1. **Pull Image** - Download Unsloth (~7GB)
        2. **Start Container** - Launch with GPU
        3. **Configure** - Select model & dataset
        4. **Train** - Click Start Training
        5. **Done** - Model in `work/models/`
        """)
    
    st.markdown("---")
    
    # Trained models section - simplified with link to inference page
    st.markdown("### TRAINED_MODELS")
    
    work_models_dir = Path("work/models")
    models_dir = Path("models")
    
    trained_models = []
    for search_dir in [work_models_dir, models_dir]:
        if search_dir.exists():
            for model_path in search_dir.iterdir():
                if model_path.is_dir() and (model_path / "adapter_config.json").exists():
                    trained_models.append(model_path)
    
    if not trained_models:
        st.info("[INFO] No trained models yet. Start training to create your first model!")
    else:
        col_list, col_action = st.columns([2, 1])
        
        with col_list:
            for i, model_path in enumerate(trained_models[:8]):
                st.markdown(f"""
                <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
                    <div style="color: white; font-weight: bold;">MODEL: {model_path.name}</div>
                    <div style="color: #8c6b5d; font-size: 0.75rem;">Path: {model_path}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_action:
            st.markdown(f"""
            <div class="industrial-plate" style="border-left: 3px solid #f96124; text-align: center;">
                <div style="color: #f96124; font-weight: bold; font-size: 1.2rem;">TEST_MODELS</div>
                <div style="color: #8c6b5d; font-size: 0.85rem; margin-top: 0.5rem;">
                    {len(trained_models)} models ready for testing
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("[GO_TO_INFERENCE]", type="primary", use_container_width=True):
                st.session_state.view = 'inference'
                st.rerun()


# ==================== VIEW: INFERENCE ====================

def render_inference():
    """Render the inference testing page with Arena mode."""
    from src.finetuning_manager import DockerFineTuningManager, TrainingConfig
    from src.backboard_manager import BackboardManager
    
    # Header with navigation
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("""
        <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
            MODEL INFERENCE
        </h1>
        <p style="color: #8c6b5d; font-size: 0.8rem;">
            Test and compare models with Unsloth 2x speed optimization
        </p>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("[DASHBOARD]", use_container_width=True):
            st.session_state.view = 'dashboard'
            st.rerun()
    with col3:
        if st.button("[FINE_TUNING]", use_container_width=True):
            st.session_state.view = 'fine_tuning'
            st.rerun()
    
    st.markdown("---")
    
    # Initialize manager
    manager = DockerFineTuningManager()
    container_running = manager.get_running_container() is not None
    
    # Container status
    if container_running:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
            <span style="color: #22c55e;">[RUNNING]</span> <strong>[CONTAINER_RUNNING]</strong> - Ready for inference
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 3px solid #ef4444;">
            <span style="color: #ef4444;">[OFFLINE]</span> <strong>[CONTAINER_OFFLINE]</strong>
            <div style="color: #8c6b5d; font-size: 0.85rem;">Go to Fine-Tuning page to start the container</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("---")
    
    # Get trained models
    work_models_dir = Path("work/models")
    models_dir = Path("models")
    
    trained_models = []
    for search_dir in [work_models_dir, models_dir]:
        if search_dir.exists():
            for model_path in search_dir.iterdir():
                if model_path.is_dir() and (model_path / "adapter_config.json").exists():
                    trained_models.append(model_path)
    
    if not trained_models:
        st.warning("[WARN] No trained models found. Train a model first on the Fine-Tuning page!")
        return
    
    # Mode selection tabs
    mode_tab = st.radio(
        "Testing Mode",
        ["[SINGLE_MODEL]", "[ARENA_MODE]"],
        horizontal=True,
        key="inference_mode"
    )
    
    st.markdown("---")
    
    if mode_tab == "[SINGLE_MODEL]":
        render_single_inference(manager, trained_models)
    else:
        render_arena_mode(manager, trained_models)


def render_single_inference(manager, trained_models):
    """Render single model inference UI."""
    # Two column layout
    col_model, col_test = st.columns([1, 2])
    
    with col_model:
        st.markdown("### SELECT_MODEL")
        
        # Model selection cards
        for i, model_path in enumerate(trained_models):
            is_selected = st.session_state.get('selected_test_model') == str(model_path)
            border_color = "#f96124" if is_selected else "#3d3d3d"
            
            st.markdown(f"""
            <div class="industrial-plate" style="border-left: 3px solid {border_color};">
                <div style="color: white; font-weight: bold;">MODEL: {model_path.name}</div>
                <div style="color: #8c6b5d; font-size: 0.75rem;">Path: {model_path}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Select", key=f"select_model_inf_{i}", use_container_width=True):
                st.session_state.selected_test_model = str(model_path)
                st.session_state.inference_result = None
                st.rerun()
    
    with col_test:
        st.markdown("### TEST_INFERENCE")
        
        selected_model = st.session_state.get('selected_test_model')
        
        if not selected_model:
            st.info("[INFO] Select a model from the left to begin testing")
            return
        
        # Selected model display
        st.markdown(f"""
        <div class="industrial-plate" style="border-left: 3px solid #f96124;">
            <div style="color: #f96124; font-size: 0.85rem;">SELECTED_MODEL:</div>
            <div style="color: white; font-weight: bold; font-size: 1.1rem;">{Path(selected_model).name}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prompt input
        st.markdown("#### ENTER_PROMPT")
        
        if 'test_prompt' not in st.session_state:
            st.session_state.test_prompt = "Hello! Can you introduce yourself and explain what you can help with?"
        
        test_prompt = st.text_area(
            "Prompt:",
            value=st.session_state.test_prompt,
            height=120,
            key="inference_prompt_input",
            placeholder="Enter your prompt here..."
        )
        
        # Check if we're currently running inference
        is_inferencing = st.session_state.get('inference_running', False)
        
        col_run, col_clear = st.columns([2, 1])
        
        with col_run:
            if st.button(
                "[RUN_INFERENCE]" if not is_inferencing else "[PROCESSING]",
                type="primary",
                use_container_width=True,
                disabled=is_inferencing or not test_prompt.strip()
            ):
                st.session_state.test_prompt = test_prompt
                st.session_state.inference_running = True
                st.session_state.inference_result = None
                st.rerun()
        
        with col_clear:
            if st.button("[CLEAR]", use_container_width=True):
                st.session_state.inference_result = None
                st.rerun()
        
        # Execute inference if flag is set
        if st.session_state.get('inference_running'):
            with st.spinner("[PROCESSING] Running inference with Unsloth 2x speed... (First load takes ~30-60 seconds)"):
                prompt_to_use = st.session_state.get('test_prompt', test_prompt)
                success, result = manager.run_inference_in_container(st.session_state.selected_test_model, prompt_to_use)
                st.session_state.inference_running = False
                st.session_state.inference_result = {
                    "success": success,
                    "output": result if result else "No output received",
                    "prompt": prompt_to_use
                }
                st.rerun()
        
        # Display inference result
        if st.session_state.get('inference_result') is not None:
            st.markdown("---")
            result = st.session_state.inference_result
            
            if result.get('success'):
                st.markdown("#### MODEL_RESPONSE")
                clean_response = extract_model_response(result.get('output', ''))
                
                st.markdown(f"""
                <div class="industrial-plate" style="border-left: 3px solid #22c55e; padding: 1rem;">
                    <div style="color: white; white-space: pre-wrap; font-family: monospace;">{clean_response}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show full output in expander
                with st.expander("[VIEW_FULL_OUTPUT]"):
                    st.code(result.get('output', ''), language="text")
            else:
                st.error(f"[FAILED] Inference failed: {result.get('output', 'Unknown error')}")


def extract_model_response(output: str) -> str:
    """Extract clean response from inference output."""
    if "MODEL RESPONSE:" in output and "-" * 60 in output:
        try:
            parts = output.split("MODEL RESPONSE:")
            if len(parts) > 1:
                response_part = parts[1].split("-" * 60)
                if len(response_part) > 1:
                    clean_response = response_part[1].strip()
                    if "Inference complete" in clean_response:
                        clean_response = clean_response.split("Inference complete")[0].strip()
                    return clean_response
        except:
            pass
    return output


def render_arena_mode(manager, trained_models):
    """Render the Arena mode for comparing fine-tuned vs base models."""
    from src.backboard_manager import BackboardManager
    import asyncio
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="color: #f96124; font-size: 1.8rem; font-weight: 900; letter-spacing: 0.15em;">
            MODEL ARENA (BATTLE)
        </h2>
        <p style="color: #8c6b5d; font-size: 0.85rem;">
            Compare your fine-tuned model against a base model from Backboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration section
    st.markdown("### ARENA_CONFIGURATION")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 3px solid #f96124;">
            <div style="color: #f96124; font-weight: bold; font-size: 0.9rem;">CHALLENGER (FT)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Fine-tuned model selection
        model_names = [p.name for p in trained_models]
        selected_finetuned = st.selectbox(
            "Select Fine-tuned Model",
            options=model_names,
            key="arena_finetuned_model"
        )
        finetuned_path = next((p for p in trained_models if p.name == selected_finetuned), None)
    
    with col_right:
        st.markdown("""
        <div class="industrial-plate" style="border-left: 3px solid #22c55e;">
            <div style="color: #22c55e; font-weight: bold; font-size: 0.9rem;">🛡️ DEFENDER (Base Model)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check API key
        api_key = st.session_state.get('api_key')
        if not api_key:
            st.warning("⚠️ Set Backboard API key in sidebar to use base models")
            base_model = None
        else:
            # Direct model input - type the model name from Backboard library
            st.caption("💡 Enter model in format: `provider/model-name`")
            st.caption("Examples: `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-20241022`")
            
            base_model = st.text_input(
                "Model Name",
                value=st.session_state.get('arena_base_model_input', 'openai/gpt-4o'),
                key="arena_base_model_input",
                placeholder="openai/gpt-4o"
            )
            
            if not base_model or '/' not in base_model:
                st.warning("⚠️ Enter model as `provider/model-name`")
                base_model = None
    
    st.markdown("---")
    
    # Judge configuration
    st.markdown("### ⚖️ Judge Configuration (Optional)")
    
    col_judge, col_criteria = st.columns([1, 2])
    
    with col_judge:
        enable_judge = st.checkbox("Enable Judge Model", key="arena_enable_judge")
        
        if enable_judge:
            # Direct judge model input
            st.caption("💡 Use a capable model like Claude or GPT-4")
            
            judge_model = st.text_input(
                "Judge Model",
                value=st.session_state.get('arena_judge_model_input', 'anthropic/claude-3-5-sonnet-20241022'),
                key="arena_judge_model_input",
                placeholder="anthropic/claude-3-5-sonnet-20241022"
            )
            
            if not judge_model or '/' not in judge_model:
                st.warning("⚠️ Enter model as `provider/model-name`")
                judge_model = None
        else:
            judge_model = None
    
    with col_criteria:
        if enable_judge:
            judge_criteria = st.text_area(
                "Judging Criteria",
                value="""Evaluate both responses based on:
1. **Accuracy** - Is the information correct and factual?
2. **Relevance** - Does it answer the question directly?
3. **Clarity** - Is the response well-structured and easy to understand?
4. **Completeness** - Does it cover all aspects of the query?
5. **Helpfulness** - How useful is the response overall?

Provide a brief analysis of each response and declare a winner.""",
                height=180,
                key="arena_judge_criteria"
            )
    
    st.markdown("---")
    
    # Prompt input
    st.markdown("### ARENA_PROMPT")
    
    arena_prompt = st.text_area(
        "Enter the prompt to test both models:",
        height=120,
        key="arena_prompt_input",
        placeholder="Enter your prompt here... Both models will receive the same prompt."
    )
    
    # Run Arena button
    col_run, col_clear = st.columns([3, 1])
    
    is_running = st.session_state.get('arena_running', False)
    
    with col_run:
        run_disabled = is_running or not arena_prompt.strip() or not base_model or not finetuned_path
        if st.button(
            "[START_BATTLE]" if not is_running else "[BATTLE_IN_PROGRESS]",
            type="primary",
            use_container_width=True,
            disabled=run_disabled
        ):
            st.session_state.arena_running = True
            st.session_state.arena_results = None
            st.session_state.arena_prompt = arena_prompt
            st.rerun()
    
    with col_clear:
        if st.button("[CLEAR]", use_container_width=True):
            st.session_state.arena_results = None
            st.session_state.judge_result = None
            st.rerun()
    
    # Execute arena battle
    if st.session_state.get('arena_running'):
        st.markdown("---")
        st.markdown("### ⚔️ Battle in Progress...")
        
        prompt_to_use = st.session_state.get('arena_prompt', arena_prompt)
        results = {}
        
        # Progress indicators
        progress_placeholder = st.empty()
        
        # Run fine-tuned model
        progress_placeholder.markdown("🔥 Running **Challenger** (Fine-tuned model)...")
        success_ft, result_ft = manager.run_inference_in_container(str(finetuned_path), prompt_to_use)
        results['finetuned'] = {
            'success': success_ft,
            'output': extract_model_response(result_ft) if success_ft else result_ft,
            'raw_output': result_ft,
            'model_name': selected_finetuned
        }
        
        # Run base model and optionally judge via Backboard (single event loop)
        progress_placeholder.markdown("🌐 Running **Defender** (Base model via Backboard)...")
        try:
            backboard_manager = BackboardManager(api_key=st.session_state.api_key)
            
            # Parse model provider/name
            model_parts = base_model.split("/")
            llm_provider = model_parts[0] if len(model_parts) > 1 else "openai"
            model_name = model_parts[1] if len(model_parts) > 1 else base_model
            
            # Parse judge model if enabled
            run_judge = enable_judge and judge_model
            if run_judge:
                judge_parts = judge_model.split("/")
                judge_provider = judge_parts[0] if len(judge_parts) > 1 else "openai"
                judge_model_name = judge_parts[1] if len(judge_parts) > 1 else judge_model
            
            # Combined async function for base model and judge
            async def run_arena_models():
                arena_results = {'base': None, 'judge': None}
                
                # Run base model
                try:
                    assistant = await backboard_manager.client.create_assistant(
                        name="Arena Base Model",
                        description="You are a helpful AI assistant. Answer the user's question directly and helpfully."
                    )
                    thread = await backboard_manager.client.create_thread(assistant.assistant_id)
                    response = await backboard_manager.client.add_message(
                        thread_id=thread.thread_id,
                        content=prompt_to_use,
                        llm_provider=llm_provider,
                        model_name=model_name,
                        stream=False
                    )
                    arena_results['base'] = {
                        'success': True,
                        'output': response.content if hasattr(response, 'content') else str(response),
                        'model_name': base_model
                    }
                except Exception as e:
                    arena_results['base'] = {
                        'success': False,
                        'output': f"Error: {str(e)}",
                        'model_name': base_model
                    }
                
                # Run judge if enabled and base succeeded
                if run_judge:
                    try:
                        finetuned_output = results['finetuned']['output']
                        base_output = arena_results['base']['output']
                        
                        judge_prompt = f"""You are an impartial judge evaluating two AI model responses.

**PROMPT GIVEN TO BOTH MODELS:**
{prompt_to_use}

**RESPONSE A (Model: {results['finetuned']['model_name']}):**
{finetuned_output}

**RESPONSE B (Model: {arena_results['base']['model_name']}):**
{base_output}

**EVALUATION CRITERIA:**
{st.session_state.get('arena_judge_criteria', 'Evaluate overall quality')}

Please provide:
1. A brief analysis of Response A
2. A brief analysis of Response B
3. Your verdict: Which response is better and why?
4. Final declaration: "WINNER: A" or "WINNER: B" or "TIE"
"""
                        
                        judge_assistant = await backboard_manager.client.create_assistant(
                            name="Arena Judge",
                            description="You are an impartial AI judge. Evaluate responses fairly and provide clear verdicts."
                        )
                        judge_thread = await backboard_manager.client.create_thread(judge_assistant.assistant_id)
                        judge_response = await backboard_manager.client.add_message(
                            thread_id=judge_thread.thread_id,
                            content=judge_prompt,
                            llm_provider=judge_provider,
                            model_name=judge_model_name,
                            stream=False
                        )
                        arena_results['judge'] = {
                            'success': True,
                            'verdict': judge_response.content if hasattr(judge_response, 'content') else str(judge_response),
                            'judge_model': judge_model
                        }
                    except Exception as e:
                        arena_results['judge'] = {
                            'success': False,
                            'verdict': f"Judge error: {str(e)}",
                            'judge_model': judge_model
                        }
                
                return arena_results
            
            # Run everything in a single event loop
            arena_outputs = asyncio.run(run_arena_models())
            results['base'] = arena_outputs['base']
            
            if arena_outputs.get('judge'):
                st.session_state.judge_result = arena_outputs['judge']
                
        except Exception as e:
            results['base'] = {
                'success': False,
                'output': f"Error: {str(e)}",
                'model_name': base_model
            }
        
        progress_placeholder.empty()
        st.session_state.arena_running = False
        st.session_state.arena_results = results
        st.rerun()
    
    # Display arena results
    if st.session_state.get('arena_results'):
        st.markdown("---")
        st.markdown("### ARENA_RESULTS")
        
        results = st.session_state.arena_results
        
        col_a, col_vs, col_b = st.columns([5, 1, 5])
        
        with col_a:
            ft_result = results.get('finetuned', {})
            success_a = ft_result.get('success', False)
            color_a = "#f96124" if success_a else "#ef4444"
            
            st.markdown(f"""
            <div class="industrial-plate" style="border: 2px solid {color_a}; min-height: 300px;">
                <div style="color: {color_a}; font-weight: bold; font-size: 1rem; margin-bottom: 0.5rem; text-align: center;">
                    CHALLENGER (FT)
                </div>
                <div style="color: #8c6b5d; font-size: 0.75rem; text-align: center; margin-bottom: 1rem;">
                    {ft_result.get('model_name', 'Unknown')}
                </div>
                <div style="color: white; white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; max-height: 400px; overflow-y: auto;">
                    {ft_result.get('output', 'No output')[:2000]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_vs:
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
                <span style="font-size: 2rem; color: #f96124; font-weight: 900;">VS</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            base_result = results.get('base', {})
            success_b = base_result.get('success', False)
            color_b = "#22c55e" if success_b else "#ef4444"
            
            st.markdown(f"""
            <div class="industrial-plate" style="border: 2px solid {color_b}; min-height: 300px;">
                <div style="color: {color_b}; font-weight: bold; font-size: 1rem; margin-bottom: 0.5rem; text-align: center;">
                    DEFENDER (BASE)
                </div>
                <div style="color: #8c6b5d; font-size: 0.75rem; text-align: center; margin-bottom: 1rem;">
                    {base_result.get('model_name', 'Unknown')}
                </div>
                <div style="color: white; white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; max-height: 400px; overflow-y: auto;">
                    {base_result.get('output', 'No output')[:2000]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # User voting
        st.markdown("---")
        st.markdown("### YOUR_VOTE")
        
        col_vote_a, col_tie, col_vote_b = st.columns(3)
        
        with col_vote_a:
            if st.button("[CHALLENGER_WINS]", use_container_width=True, type="secondary"):
                st.session_state.arena_user_vote = "challenger"
                st.success("[SUCCESS] You voted for the Challenger (Fine-tuned model)!")
        
        with col_tie:
            if st.button("[TIE]", use_container_width=True):
                st.session_state.arena_user_vote = "tie"
                st.info("🤝 You declared a tie!")
        
        with col_vote_b:
            if st.button("[DEFENDER_WINS]", use_container_width=True, type="secondary"):
                st.session_state.arena_user_vote = "defender"
                st.success("[SUCCESS] You voted for the Defender (Base model)!")
        
        # Display judge verdict if available
        if st.session_state.get('judge_result'):
            st.markdown("---")
            st.markdown("### JUDGE_VERDICT")
            
            judge_result = st.session_state.judge_result
            
            if judge_result.get('success'):
                verdict = judge_result.get('verdict', '')
                
                # Determine winner highlight
                winner_color = "#f96124"  # Default
                if "WINNER: A" in verdict:
                    winner_text = "[CHALLENGER_WINS!]"
                    winner_color = "#f96124"
                elif "WINNER: B" in verdict:
                    winner_text = "[DEFENDER_WINS!]"
                    winner_color = "#22c55e"
                else:
                    winner_text = "[TIE]"
                    winner_color = "#f59e0b"
                
                st.markdown(f"""
                <div class="industrial-plate" style="border: 2px solid {winner_color};">
                    <div style="color: {winner_color}; font-weight: bold; font-size: 1.2rem; text-align: center; margin-bottom: 1rem;">
                        {winner_text}
                    </div>
                    <div style="color: #8c6b5d; font-size: 0.75rem; text-align: center; margin-bottom: 1rem;">
                        Judged by: {judge_result.get('judge_model', 'Unknown')}
                    </div>
                    <div style="color: white; white-space: pre-wrap; font-size: 0.9rem;">
                        {verdict}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"[ERROR] Judge error: {judge_result.get('verdict', 'Unknown error')}")
        
        # Full outputs in expanders
        st.markdown("---")
        with st.expander("[VIEW_CHALLENGER_OUTPUT]"):
            st.code(results.get('finetuned', {}).get('raw_output', results.get('finetuned', {}).get('output', '')), language="text")
        
        with st.expander("[VIEW_DEFENDER_OUTPUT]"):
            st.code(results.get('base', {}).get('output', ''), language="text")


# ==================== VIEW: HUGGINGFACE DEPLOY ====================

def render_hf_deploy():
    """Render the HuggingFace deployment page for uploading models and datasets."""
    
    # Header with navigation
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <h1 style="color: white; font-size: 2rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em;">
            🤗 HUGGINGFACE DEPLOY
        </h1>
        <p style="color: #8c6b5d; font-size: 0.8rem;">
            Upload your fine-tuned models and datasets to HuggingFace Hub
        </p>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("[DASHBOARD]", use_container_width=True):
            st.session_state.view = 'dashboard'
            st.rerun()
    
    st.markdown("---")
    
    # HuggingFace Token Setup
    st.markdown("### 🔑 AUTHENTICATION")
    
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        value=st.session_state.get('hf_token', ''),
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Get your token from https://huggingface.co/settings/tokens"
    )
    if hf_token:
        st.session_state.hf_token = hf_token
        st.success("✓ Token configured")
    else:
        st.warning("⚠️ Enter your HuggingFace token to enable uploads")
        st.markdown("""
        <div style="background: #1a1008; border-left: 3px solid #ff6b35; padding: 0.75rem; margin: 0.5rem 0; font-size: 0.85rem;">
            <strong>Get a token:</strong> Visit <a href="https://huggingface.co/settings/tokens" target="_blank" style="color: #f96124;">huggingface.co/settings/tokens</a>
            and create a token with <code>write</code> permissions.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two columns: Models and Datasets
    col_models, col_datasets = st.columns(2)
    
    # ==================== MODEL UPLOAD ====================
    with col_models:
        st.markdown("### 🧠 UPLOAD MODEL")
        
        # Scan for available models
        models_dir = Path("work/models")
        available_models = []
        if models_dir.exists():
            for model_path in models_dir.iterdir():
                if model_path.is_dir() and (model_path / "adapter_config.json").exists():
                    available_models.append(model_path.name)
                elif model_path.is_dir() and (model_path / "config.json").exists():
                    available_models.append(model_path.name)
        
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                key="hf_model_select"
            )
            
            # Repository settings
            st.markdown("##### Repository Settings")
            
            hf_username = st.text_input(
                "HuggingFace Username",
                value=st.session_state.get('hf_username', ''),
                placeholder="your-username",
                key="hf_username_model"
            )
            if hf_username:
                st.session_state.hf_username = hf_username
            
            repo_name = st.text_input(
                "Repository Name",
                value=selected_model if selected_model else "",
                placeholder="my-awesome-model",
                key="hf_repo_name_model"
            )
            
            private_repo = st.checkbox("Private Repository", value=False, key="hf_private_model")
            
            # Model card
            with st.expander("📝 Model Card (Optional)"):
                model_description = st.text_area(
                    "Description",
                    placeholder="A fine-tuned model for...",
                    height=100,
                    key="hf_model_desc"
                )
            
            # Upload button
            if st.button("🚀 UPLOAD MODEL", type="primary", use_container_width=True, key="upload_model_btn"):
                if not hf_token:
                    st.error("[ERROR] Please enter your HuggingFace token")
                elif not hf_username:
                    st.error("[ERROR] Please enter your HuggingFace username")
                elif not repo_name:
                    st.error("[ERROR] Please enter a repository name")
                else:
                    with st.spinner(f"Uploading {selected_model} to HuggingFace..."):
                        try:
                            from huggingface_hub import HfApi, create_repo
                            
                            api = HfApi(token=hf_token)
                            repo_id = f"{hf_username}/{repo_name}"
                            
                            # Create repo if it doesn't exist
                            try:
                                create_repo(repo_id, token=hf_token, private=private_repo, exist_ok=True)
                            except Exception as e:
                                if "already exists" not in str(e).lower():
                                    raise e
                            
                            # Upload folder
                            model_path = models_dir / selected_model
                            api.upload_folder(
                                folder_path=str(model_path),
                                repo_id=repo_id,
                                commit_message=f"Upload {selected_model} via ModelForge"
                            )
                            
                            st.success(f"✓ Model uploaded successfully!")
                            st.markdown(f"""
                            <div style="background: #0a2810; border: 1px solid #22c55e; padding: 1rem; margin-top: 0.5rem;">
                                <strong>🎉 Your model is live!</strong><br>
                                <a href="https://huggingface.co/{repo_id}" target="_blank" style="color: #22c55e;">
                                    https://huggingface.co/{repo_id}
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"[ERROR] Upload failed: {str(e)}")
        else:
            st.info("📦 No models found in `work/models/`. Train a model first!")
            if st.button("[GO_TO_TRAINING]", use_container_width=True):
                st.session_state.view = 'fine_tuning'
                st.rerun()
    
    # ==================== DATASET UPLOAD ====================
    with col_datasets:
        st.markdown("### 📊 UPLOAD DATASET")
        
        # Scan for available datasets
        data_dir = Path("data")
        available_datasets = []
        if data_dir.exists():
            for file_path in data_dir.iterdir():
                if file_path.suffix in ['.jsonl', '.json', '.csv', '.parquet']:
                    available_datasets.append(file_path.name)
        
        if available_datasets:
            selected_dataset = st.selectbox(
                "Select Dataset",
                available_datasets,
                key="hf_dataset_select"
            )
            
            # Show dataset preview
            if selected_dataset:
                dataset_path = data_dir / selected_dataset
                try:
                    if selected_dataset.endswith('.jsonl'):
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:5]
                        st.caption(f"Preview ({len(lines)} of {sum(1 for _ in open(dataset_path, 'r', encoding='utf-8'))} samples):")
                        for line in lines:
                            st.json(json.loads(line))
                    elif selected_dataset.endswith('.json'):
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            st.caption(f"Preview (first 3 of {len(data)} samples):")
                            for item in data[:3]:
                                st.json(item)
                except Exception as e:
                    st.warning(f"Could not preview: {e}")
            
            # Repository settings
            st.markdown("##### Repository Settings")
            
            hf_username_ds = st.text_input(
                "HuggingFace Username",
                value=st.session_state.get('hf_username', ''),
                placeholder="your-username",
                key="hf_username_dataset"
            )
            if hf_username_ds:
                st.session_state.hf_username = hf_username_ds
            
            dataset_repo_name = st.text_input(
                "Repository Name",
                value=selected_dataset.rsplit('.', 1)[0] if selected_dataset else "",
                placeholder="my-awesome-dataset",
                key="hf_repo_name_dataset"
            )
            
            private_ds_repo = st.checkbox("Private Repository", value=False, key="hf_private_dataset")
            
            # Upload button
            if st.button("🚀 UPLOAD DATASET", type="primary", use_container_width=True, key="upload_dataset_btn"):
                if not hf_token:
                    st.error("[ERROR] Please enter your HuggingFace token")
                elif not hf_username_ds:
                    st.error("[ERROR] Please enter your HuggingFace username")
                elif not dataset_repo_name:
                    st.error("[ERROR] Please enter a repository name")
                else:
                    with st.spinner(f"Uploading {selected_dataset} to HuggingFace..."):
                        try:
                            from huggingface_hub import HfApi, create_repo
                            
                            api = HfApi(token=hf_token)
                            repo_id = f"{hf_username_ds}/{dataset_repo_name}"
                            
                            # Create dataset repo
                            try:
                                create_repo(repo_id, token=hf_token, private=private_ds_repo, 
                                           repo_type="dataset", exist_ok=True)
                            except Exception as e:
                                if "already exists" not in str(e).lower():
                                    raise e
                            
                            # Upload file
                            dataset_path = data_dir / selected_dataset
                            api.upload_file(
                                path_or_fileobj=str(dataset_path),
                                path_in_repo=selected_dataset,
                                repo_id=repo_id,
                                repo_type="dataset",
                                commit_message=f"Upload {selected_dataset} via ModelForge"
                            )
                            
                            st.success(f"✓ Dataset uploaded successfully!")
                            st.markdown(f"""
                            <div style="background: #0a2810; border: 1px solid #22c55e; padding: 1rem; margin-top: 0.5rem;">
                                <strong>🎉 Your dataset is live!</strong><br>
                                <a href="https://huggingface.co/datasets/{repo_id}" target="_blank" style="color: #22c55e;">
                                    https://huggingface.co/datasets/{repo_id}
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"[ERROR] Upload failed: {str(e)}")
        else:
            st.info("📦 No datasets found in `data/`. Generate some data first!")
            if st.button("[GO_TO_DASHBOARD]", use_container_width=True):
                st.session_state.view = 'dashboard'
                st.rerun()
    
    # Tips section
    st.markdown("---")
    st.markdown("### 💡 TIPS")
    st.markdown("""
    <div style="background: #111; border: 1px solid #333; padding: 1rem; font-size: 0.85rem;">
        <ul style="margin: 0; padding-left: 1.5rem; color: #888;">
            <li><strong>Model uploads</strong> include all adapter files (LoRA weights, config, tokenizer)</li>
            <li><strong>Dataset uploads</strong> preserve the original format (JSONL, JSON, CSV, Parquet)</li>
            <li>Use <strong>private repositories</strong> for proprietary data or models</li>
            <li>After upload, you can load models with: <code>model = AutoModel.from_pretrained("username/repo")</code></li>
            <li>After upload, you can load datasets with: <code>dataset = load_dataset("username/repo")</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
