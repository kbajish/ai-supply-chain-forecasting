import pandas as pd
import numpy as np
from pathlib import Path


def test_demand_data_generated():
    from src.data.generator import generate_demand_data
    df = generate_demand_data()
    assert len(df) == 36500
    assert "product_id" in df.columns
    assert "demand" in df.columns
    assert "date" in df.columns


def test_sku_format():
    from src.data.generator import generate_demand_data
    df = generate_demand_data()
    for sku in df["product_id"].unique():
        assert sku.startswith("SKU-")


def test_demand_non_negative():
    from src.data.generator import generate_demand_data
    df = generate_demand_data()
    assert (df["demand"] >= 0).all()


def test_supplier_risk_generated():
    from src.data.generator import generate_supplier_risk
    df = generate_supplier_risk()
    assert len(df) == 100
    assert "supplier_id" in df.columns
    assert "risk_level" in df.columns


def test_supplier_id_format():
    from src.data.generator import generate_supplier_risk
    df = generate_supplier_risk()
    for sid in df["supplier_id"]:
        assert sid.startswith("SUP-")


def test_risk_levels_valid():
    from src.data.generator import generate_supplier_risk
    df = generate_supplier_risk()
    assert set(df["risk_level"].unique()).issubset({"Low", "Medium", "High"})


def test_currency_features():
    from src.data.generator import generate_demand_data
    df = generate_demand_data()
    assert "day_of_week" in df.columns
    assert "month" in df.columns
    assert "is_weekend" in df.columns