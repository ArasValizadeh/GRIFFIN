# GRIFFIN

GRIFFIN is a PyTorch-based neuro-fuzzy inference system designed specifically for classification tasks.
It builds upon the UNFIS framework but introduces several key enhancements:
	•	Feature Selection – automatically learns which input features are most relevant.
	•	Relaxation Mechanism – smooths rule activations for improved generalization.
	•	Literal Modulation – adds flexibility in handling fuzzy membership functions.
	•	Correlation-Aware Design – captures dependencies among input features for more expressive rules.

By combining these mechanisms, GRIFFIN provides a robust and interpretable alternative to black-box neural models in classification problems.


## 📂 Project Structure
GRIFFIN/
├── GRIFFIN/
│   ├── model/                # Core model implementation
│   │   └── griffin.py
│   ├── train/                # Training utilities
│   │   ├── early_stop.py
│   │   └── classification_test.py
│   ├── tests/                # Unit tests
│   │   ├── class_test.py
│   │   ├── uci_test.py
│   │   └── init.py
│   └── notebooks/            # Experiment notebooks
│       └── test_class.ipynb
└── data/                     # Datasets (excluded from repo)

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/GRIFFIN.git
   cd GRIFFIN

2.	Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate   # on Linux/Mac
    .venv\Scripts\activate      # on Windows

3.	Install dependencies:
    pip install -r requirements.txt

📁 Data

Datasets are expected in a data/ directory at the project root.
This folder is not included in the repository due to size.
Place your datasets there before running training/testing.

🧪 Tests

Unit tests are located in GRIFFIN/tests/.
They include both classification and UCI dataset examples.

📜 License

MIT License – feel free to use and adapt.