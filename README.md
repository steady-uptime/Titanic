# Production-Grade MLOps Pipeline: Titanic Dataset Refactoring

This repository demonstrates a modular, production-ready MLOps architecture designed for scalability, portability, and maintainability. Unlike standard "research" notebooks, this project follows strict Software Engineering principles to ensure the transition from model development to production deployment is seamless.

## 🏗 Architecture Principles
To ensure production readiness, this project adheres to the following engineering constraints:
- **Config-Driven Architecture**: Zero hardcoding. All hyperparameters, file paths, and transformation rules are externalized in `configs/`.
- **Decoupled Logic**: Separation of Concerns (SoC) between Data Engineering (ETL), Model Engineering (Training/Inference), and Orchestration (Workflow Management).
- **Portability**: All paths are resolved dynamically via a Singleton Configuration Loader using `pathlib`, ensuring the project runs on any OS without modification.
- **Artifact Immutability**: Data transformations produce new artifacts rather than modifying source data in place.

## 🚀 Project Status
- [x] **Phase 1: Configuration Engine** (Singleton Config Loader, Profile Support)
- [x] **Phase 2: Data Engineering Pipeline** (Worker-based ETL, Dynamic Preprocessing)
- [x] **Phase 3: Model Engineering** (Trainable Workers, Artifact Management) - *In Progress*
- [ ] **Phase 4: Evaluation & Orchestration** (Metrics, Logging, Scheduling)
- [ ] **Phase 5: Deployment** (Docker, Kubernetes, API)

## 🚀 Key Achievements
- Singleton Pattern for Configuration Management.
- Abstract Base Classes (ABC) for Model Contract Definition.
- Dependency Injection for Preprocessing Rules.
- Contract Validation for Data Integrity.

## 🛠 Component Breakdown

### 1. Configuration Management (`configs/`)
- **Singleton Pattern**: The `src/config/config_loader.py` ensures that the configuration is loaded into memory once. This provides a **Single Source of Truth** and prevents redundant I/O operations across the application.
- **Profile Support**: Support for `development`, `staging`, and `production` profiles within `config.yaml`.

### 2. Data Engineering (`src/data/`)
- **Worker Pattern**: Data loading and preprocessing are abstracted into dedicated worker classes.
- **Dynamic Preprocessing**: The pipeline handles dropping, imputation, and encoding via configuration injection.
- **Dependency Injection**: Preprocessing methods receive specific configuration slices, making the logic easily testable and modular.

### 3. Model Engineering (`src/model/`)
- **Abstract Base Classes (ABC)**: Defines a strict contract for all model workers (e.g., `.fit()`, `.predict()`).
- **Persistence Layer**: Standardized serialization of model artifacts using `joblib`.

### 4. Orchestration (`scripts/`)
- **Manager Pattern**: Orchestrator scripts manage the lifecycle of the data flow, acting as the "Manager" that delegates tasks to "Worker" modules.

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
```

### 3. Execution Pipeline

The project uses an orchestrator pattern. Execute the stages sequentially:

##### **Data Engineering:** 
`python scripts/run_preprocessing.py`

##### **Model Training:** 
_(Pending Phase 3 Completion)_ 
`python scripts/train.py`

## 🐳 Infrastructure

- **Docker**: Containerized environment definition in `/docker`.
- **Kubernetes**: Manifests for deployment orchestration in `/k8s`.

## 🧪 Testing & Quality Assurance

- **Unit/Integration Tests**: `pytest tests/`
- **Logging**: Centralized logging routed to the `logs/` directory with `loguru` support.