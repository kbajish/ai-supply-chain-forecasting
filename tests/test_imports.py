def test_generator_imports():
    from src.data.generator import generate_demand_data, generate_supplier_risk
    assert callable(generate_demand_data)
    assert callable(generate_supplier_risk)


def test_preprocess_imports():
    from src.data.preprocess import load_demand, add_features, load_risk
    assert callable(load_demand)
    assert callable(add_features)
    assert callable(load_risk)


def test_predictor_imports():
    from src.forecasting.predictor import forecast
    assert callable(forecast)


def test_scorer_imports():
    from src.risk.scorer import score_supplier, get_high_risk_suppliers
    assert callable(score_supplier)
    assert callable(get_high_risk_suppliers)


def test_insight_chain_imports():
    from src.llm.insight_chain import build_chain, generate_insight
    assert callable(build_chain)
    assert callable(generate_insight)


def test_products_count():
    from src.data.generator import PRODUCTS
    assert len(PRODUCTS) == 50


def test_suppliers_count():
    from src.data.generator import SUPPLIERS
    assert len(SUPPLIERS) == 10