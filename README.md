# Credit Risk Model

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Mode for Bati bank.

## Features

- **EDA Notebook**: 



## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml   
├── data/                       
│   ├── raw/                   
│   └── processed/             
├── notebooks/
│   └── eda.ipynb         
├── src/
│   ├── __init__.py
│   ├── data_processing.py     
│   ├── train.py               
│   ├── predict.py             
│   └── api/
│       ├── main.py            
│       └── pydantic_models.py 
├── tests/
│   └── test_data_processing.py  
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zerubabelalpha/credit-risk-model
   ```

   go to the dir
   ```
   cd credit-risk-model
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Example


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Credit Scoring Business Understanding

1) Basel II (and subsequent supervisory guidance) makes banks responsible for credible internal risk estimates and requires robust model governance: conceptual soundness, independent validation, ongoing monitoring, and clear documentation of model design, data, assumptions and use. In practice this means regulators and supervisors expect models used for credit risk or regulatory capital to be auditable and explainable — not black boxes — so that errors, biases or changes in performance can be detected and addressed. 


2) Why a proxy is needed: when a true, observed “default” outcome is absent or incomplete (e.g., only approved loans are visible, or time horizon is too short), you must create a proxy (delinquency >X days, charge-off, treatment flag, or behavioral trigger) so you can train a supervised model. The World Bank and empirical studies emphasize that many practical scoring systems rely on such operational proxies when full default history is unavailable. 

Business risks of using a proxy:

Label error / misclassification: the proxy may systematically mislabel good/bad outcomes (e.g., short-term delinquency ≠ eventual default), which biases PD estimates. 

Targeting & economic risk: decisions (approve/price/limit) based on a noisy proxy can increase credit losses or shrink revenue (wrong accept/reject/pricing decisions).

Regulatory & governance risk: supervisors may challenge models whose proxies are weak, opaque, or insufficiently conservative.


3) Key trade-offs: simple interpretable model (Logistic + WoE) vs complex high-performance model (Gradient Boosting) in a regulated context

Interpretability & governance

Simple (Logistic + WoE): Highly interpretable, easier to document/explain, simpler to validate and monitor; aligns well with regulatory expectations for transparency and auditability. Faster to implement governance, scorecard conversion, and manual overrides. 

Complex (GBM): Often higher raw predictive power (better separation/AUC) but harder to explain at feature-level; needs additional interpretability tools (SHAP, PDP, surrogate models) and stronger validation to satisfy supervisors.

###  Project for KAIM 