# Titanic ML Pipeline: Production-Grade MLOps Architecture

![Project Status](https://img.shields.io/badge/Status-Production_Ready-blueviolet?style=for-the-badge)

This repository demonstrates a modular, production-ready MLOps architecture designed for scalability, portability, and maintainability. It moves beyond standard research notebooks by implementing strict **Software Engineering** principles, ensuring a seamless transition from model development to production deployment.

## 🏗 Architecture Principles

To ensure production readiness, this project adheres to five core engineering constraints:

1. **Strongly-Typed Configuration Contracts**: Zero hardcoding. All hyperparameters, file paths, and transformation rules are externalized in `configs/` and mapped to **Dataclasses**. This enforces a "Fail-Fast" mechanism where configuration errors are caught during the system's boot phase, not at runtime.
2. **Decoupled Logic (Separation of Concerns)**: Strict separation between Data Engineering (ETL), Model Engineering (Training/Inference), and Orchestration (Workflow Management). Components interact via defined interfaces, not shared state.
3. **Portability & Dynamic Resolution**: All filesystem paths are resolved dynamically via a **Singleton Configuration Loader** using `pathlib`. This ensures the project is environment-agnostic and runs on any OS without modification to the source code.
4. **Dynamic Contract Validation**: A dedicated validation layer enforces data integrity by validating the **runtime-generated schema** emitted by the Feature Engineering worker. This ensures the "Data Contract" is honored before data is handed off to the Model Worker.
5. **Observability & Traceability**: Centralized logging using `loguru` for structured, leveled telemetry. Automated metadata serialization ensures experiment lineage (hyperparameters, metrics, and model URIs) is captured for every run.

## 🚀 Project Status

- [x] **Phase 1: Configuration Engine** (Singleton Config Loader, Dataclass Schema Enforcement)
- [x] **Phase 2: Data Engineering Pipeline** (Worker-based ETL, Dynamic Preprocessing)
- [x] **Phase 3: Model Engineering** (Trainable Workers, Artifact Management)
- [x] **Phase 4: Evaluation & Persistence** (Metrics, Logging, Metadata Serialization)
- [ ] **Phase 5: Deployment** (Docker, Kubernetes, API)

## 🛠 Key Engineering Achievements

- **Singleton Pattern**: Implemented for Configuration Management to provide a **Single Source of Truth** and prevent redundant I/O operations.
- **Abstract Base Classes (ABC)**: Defines a strict contract for all Model Workers, ensuring interchangeability of different ML algorithms (Polymorphism).
- **Dependency Injection (DI)**: Preprocessing rules and persistence engines are injected into workers, decoupling the "How" (logic) from the "What" (configuration).
- **Contract Validation**: A dedicated validation layer ensures the data "contract" is honored at every gate in the pipeline, preventing silent failures and downstream errors.
- **Serialization Gate**: Developed a persistence engine that handles the automatic conversion of complex configuration objects into serializable JSON, ensuring metadata integrity.
- **Immutability**: Strict adherence to data immutability by working on DataFrame copies, preventing unintended side effects in the orchestration flow.

## ⚙️ Component Breakdown

### 1. Configuration Management (`src/config/`)
- **DataClasses**: Provides type safety and schema enforcement for all configuration parameters.
- **Singleton Loader**: Dynamically calculates the `project_root` and maps YAML files to Dataclasses, providing a centralized, thread-safe configuration object.

### 2. Data Engineering (`src/data_processing/`)
- **Worker Pattern**: Data loading and preprocessing are abstracted into dedicated worker classes.
- **Feature Engineering**: Isolated module for extracting features and encoding categorical variables.
- **Validator Module**: A generic, immutable utility that performs **Source Contract Validation** (verifying raw data integrity) and **Transformation Contract Validation** (verifying engineered features) at every gate.

### 3. Model Engineering (`src/model/`)
- **Contract Definition**: Uses ABCs to ensure every model implementation supports a standardized `.train()`, `.predict()`, and `.save()` interface.
- **Persistence Layer**: Standardized serialization using `joblib` for model weights and `json` for evaluation metadata.

### 4. Orchestration (`scripts/`)
- **Orchestrator Pattern**: `scripts/train.py` acts as the primary entry point, delegating tasks to specific modules. It manages the workflow topology without knowing the internal implementation details of the workers.

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

The project uses an orchestrator pattern. Execute the training lifecycle:

##### **Training & Orchestration:**
`python scripts/train.py`


## 🐳 Infrastructure

- **Docker**: Containerized environment definition in `/docker`.
- **Kubernetes**: Manifests for deployment orchestration in `/k8s`.
- **Deployment Config**: Infrastructure-specific settings in `deployment.yaml`.

## 🧪 Testing & Quality Assurance

- **Unit/Integration Tests**: `pytest tests/`
- **Logging**: Centralized logging routed to the `logs/` directory with `loguru` support for structured, leveled telemetry.