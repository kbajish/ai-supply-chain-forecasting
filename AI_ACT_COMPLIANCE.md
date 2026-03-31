# EU AI Act & GDPR Compliance Notes

## Risk Classification
This system provides AI-generated demand forecasts and supplier risk scores
to support supply chain planning decisions. It is intended as a
decision-support tool, not an autonomous decision-maker. Under the EU AI Act,
this system is classified as limited risk.

## Transparency Measures
- All forecasts include prediction intervals — uncertainty is communicated
- SHAP values explain every supplier risk decision
- LLM narratives are grounded in model outputs, not hallucinated
- MLflow tracks all training runs with metrics and parameters

## Data Handling
- Training data is synthetically generated for demonstration purposes
- No real customer, financial, or personally identifiable data is used
- Synthetic supplier names are modelled after publicly known German automotive
  companies for realism, but do not represent actual supplier data

## Model Governance
- MLflow tracks all training runs with metrics and parameters
- Model version is recorded in every inference response
- SHAP explainability is provided for every risk classification decision

## GDPR Measures
- No personal data is collected or processed by this system
- All data inputs and outputs are operational supply chain metrics
- No user data is stored beyond the current session
