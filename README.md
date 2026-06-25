# Titanic ML Pipeline: Production-Grade MLOps Architecture

![Project Status](https://img.shields.io/badge/Status-Production_Ready-blueviolet?style=for-the-badge)

This repository demonstrates a modular, production-ready MLOps architecture designed for scalability, portability, and maintainability. Unlike standard "research" notebooks, this project follows strict **Software Engineering** principles to ensure a seamless transition from model development to production deployment.

## 🏗 Architecture Principles

To ensure production readiness, this project adheres to five core engineering constraints:

1. **Config-Driven Architecture**: Zero hardcoding. All hyperparameters, file paths, and transformation rules are externalized in `configs/`.
2. **Decoupled Logic (Separation of Concerns)**: Distinct separation between Data Engineering (ETL), Model Engineering (Training/Inference), and Orchestration (Workflow Management).
3. **Portability**: All paths are resolved dynamically via a **Singleton Configuration Loader** using `pathlib`, ensuring the project runs on any OS without modification.
4. **Dynamic Contract Validation**: A dedicated validation layer enforces data integrity by validating the **runtime-generated schema** emitted by the Feature Engineering worker. This prevents silent failures and ensures the "Data Contract" is honored before the data is handed off to the Model Worker.
5. **Observability**: Centralized logging using `loguru` to provide structured, leveled, and persistent telemetry across all system components.

## 🚀 Project Status

- [x] **Phase 1: Configuration Engine** (Singleton Config Loader, Profile Support)
- [x] **Phase 2: Data Engineering Pipeline** (Worker-based ETL, Dynamic Preprocessing)
- [x] **Phase 3: Model Engineering** (Trainable Workers, Artifact Management)
- [x] **Phase 4: Evaluation & Persistence** (Metrics, Logging, Metadata Serialization)
- [ ] **Phase 5: Deployment** (Docker, Kubernetes, API)

## 🛠 Key Engineering Achievements

- **Singleton Pattern**: Used for Configuration Management to provide a **Single Source of Truth** and prevent redundant I/O operations.
- **Abstract Base Classes (ABC)**: Defines a strict contract for all Model Workers, ensuring interchangeability of different ML algorithms (Polymorphism).
- **Dependency Injection**: Preprocessing rules and persistence engines are injected into workers, decoupling the "How" (logic) from the "What" (configuration).
- **Contract Validation**: A dedicated validation layer ensures the data "contract" is honored before data is handed off to the model worker, preventing silent failures and downstream errors.
- **Stateless vs. Stateful Transformations**: Architectural awareness of data leakage by isolating stateless transformations from split-dependent stateful logic.
- **Immutability**: Strict adherence to data immutability by working on DataFrame copies, preventing unintended side effects in the orchestration flow.

## ⚙️ Component Breakdown

### 1. Configuration Management (`configs/`)
- **Single Source of Truth**: All variables (Data, Model, Pipeline) live in `config.yaml`.
- **Dynamic Resolution**: The `src/config/config_loader.py` calculates the project root dynamically, eliminating risks associated with absolute paths.

### 2. Data Engineering (`src/data_processing/`)
- **Worker Pattern**: Data loading and preprocessing are abstracted into dedicated worker classes.
- **Feature Engineering**: Isolated module for extracting features and encoding categorical variables.
- **Validator Module**: A generic, immutable utility that performs **Source Contract Validation** (verifying raw data integrity) and **Transformation Contract Validation** (verifying engineered features) at every gate in the pipeline.
- **Immutability**: Processes data into a dedicated `data/processed/` directory to preserve the integrity of `data/raw/`.

### 3. Model Engineering (`src/model/`)
- **Contract Definition**: Uses ABCs to ensure every model implementation supports a standardized `.train()`, `.predict()`, and `.save()` interface.
- **Persistence Layer**: Standardized serialization using `joblib` and `json` for model weights and evaluation metadata.

### 4. Orchestration (`scripts/`)
- **Orchestrator Pattern**: `scripts/train.py` acts as the primary entry point, delegating tasks to the specific modules in `src/`. It manages the workflow topology without knowing the internal implementation details of the workers.

## ⚙️ Development Setup

### 1. Environment Variables
Set the environment variables to allow the `ConfigLoader` to resolve the project root and active profile.

> [!Windows]
> ```powershell
> $env:PYTHONPATH = "."
> $env:CONFIG_PROFILE = "development"
> ```

> [!Linux]
> ```bash
> export PYTHONPATH=.
> export CONFIG_PROFILE=development
> ```

### 2. Virtual Environment
```bash
# Create the virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### 3. Execution Pipeline

The project uses an orchestrator pattern. Execute the training lifecycle:

##### **Training & Orchestration:**
`python scripts/train.py`
```

## 🐳 Infrastructure

- **Docker**: Containerized environment definition in `/docker`.
- **Kubernetes**: Manifests for deployment orchestration in `/k8s`.
- **Deployment Config**: Infrastructure-specific settings in `deployment.yaml`.

## 🧪 Testing & Quality Assurance

- **Unit/Integration Tests**: `pytest tests/`
- **Logging**: Centralized logging routed to the `logs/` directory with `loguru` support for structured, leveled telemetry.