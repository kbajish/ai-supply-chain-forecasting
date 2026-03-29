# EU AI Act & DSGVO Compliance Notes

## Risk classification
This system provides AI-generated demand forecasts and supplier risk scores
to support supply chain planning decisions. It is intended as a
decision-support tool, not an autonomous decision-maker. Under the EU AI Act,
this system is classified as limited risk.

## Transparency measures
- All forecasts include prediction intervals — uncertainty is communicated
- SHAP values explain every supplier risk decision
- LLM narratives are grounded in model outputs, not hallucinated

## Data handling
- No real supplier, customer, or financial data is used
- All data is synthetically generated for demonstration purposes

## Model governance
- MLflow tracks all training runs with metrics and parameters
- Model version is recorded in every inference response