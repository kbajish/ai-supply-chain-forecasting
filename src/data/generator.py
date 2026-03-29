import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

DEMAND_DIR = Path("data/demand")
RISK_DIR   = Path("data/risk")

np.random.seed(42)

PRODUCTS = [
    {"id": f"SKU-{i:04d}", "name": name, "category": cat}
    for i, (name, cat) in enumerate([
        ("Brake Caliper Assembly",      "Brake Systems"),
        ("ABS Sensor Module",           "Sensors"),
        ("Engine Control Unit",         "Electronics"),
        ("Hydraulic Pump",              "Hydraulics"),
        ("Transmission Gear Set",       "Drivetrain"),
        ("Fuel Injection Nozzle",       "Fuel Systems"),
        ("Alternator Assembly",         "Electrical"),
        ("Steering Rack",               "Steering"),
        ("Suspension Strut",            "Suspension"),
        ("Turbocharger Unit",           "Engine"),
        ("Clutch Disc Set",             "Drivetrain"),
        ("Oil Pressure Sensor",         "Sensors"),
        ("Radiator Assembly",           "Cooling"),
        ("Exhaust Manifold",            "Exhaust"),
        ("Power Steering Pump",         "Steering"),
        ("Wheel Bearing Kit",           "Suspension"),
        ("Camshaft Sensor",             "Sensors"),
        ("Ignition Coil Pack",          "Electrical"),
        ("Air Mass Sensor",             "Sensors"),
        ("Timing Chain Kit",            "Engine"),
        ("Brake Disc Rotor",            "Brake Systems"),
        ("Shock Absorber",              "Suspension"),
        ("Water Pump Assembly",         "Cooling"),
        ("Drive Shaft",                 "Drivetrain"),
        ("Fuel Pressure Regulator",     "Fuel Systems"),
        ("Lambda Oxygen Sensor",        "Sensors"),
        ("Starter Motor",               "Electrical"),
        ("Coolant Temperature Sensor",  "Sensors"),
        ("Differential Assembly",       "Drivetrain"),
        ("Intercooler Unit",            "Engine"),
        ("Brake Master Cylinder",       "Brake Systems"),
        ("EGR Valve",                   "Engine"),
        ("Throttle Body",               "Fuel Systems"),
        ("CV Joint Boot Kit",           "Drivetrain"),
        ("Catalytic Converter",         "Exhaust"),
        ("Power Window Motor",          "Electrical"),
        ("Cabin Air Filter",            "HVAC"),
        ("AC Compressor",               "HVAC"),
        ("Wiper Motor Assembly",        "Electrical"),
        ("Glow Plug Set",               "Engine"),
        ("Fuel Pump Module",            "Fuel Systems"),
        ("Anti-Roll Bar Link",          "Suspension"),
        ("Crankshaft Position Sensor",  "Sensors"),
        ("Headlight Assembly",          "Lighting"),
        ("Rear Axle Beam",              "Suspension"),
        ("Pressure Plate Assembly",     "Drivetrain"),
        ("Knock Sensor",                "Sensors"),
        ("Mass Airflow Sensor",         "Sensors"),
        ("Vacuum Pump",                 "Engine"),
        ("Parking Brake Cable",         "Brake Systems"),
    ], start=1)
]

SUPPLIERS = [
    "Bosch Automotive GmbH",
    "ZF Friedrichshafen AG",
    "Continental AG",
    "Schaeffler Group",
    "Mahle GmbH",
    "Brose Fahrzeugteile",
    "Hella GmbH",
    "Webasto Group",
    "Knorr-Bremse AG",
    "Valeo Deutschland GmbH",
]


def generate_demand_data() -> pd.DataFrame:
    log.info("Generating demand time series for 50 SKUs...")
    dates   = pd.date_range(start="2023-01-01", periods=730, freq="D")
    records = []

    for product in PRODUCTS:
        base_demand = np.random.randint(20, 200)
        trend       = np.linspace(0, np.random.uniform(-0.1, 0.3), 730)
        weekly_s    = 10 * np.sin(2 * np.pi * np.arange(730) / 7)
        monthly_s   = 15 * np.sin(2 * np.pi * np.arange(730) / 30)
        noise       = np.random.normal(0, base_demand * 0.1, 730)

        # Supply disruption events
        disruptions = np.zeros(730)
        for _ in range(np.random.randint(2, 6)):
            start = np.random.randint(0, 700)
            disruptions[start:start+14] += np.random.uniform(0.3, 0.8) * base_demand

        demand = base_demand * (1 + trend) + weekly_s + monthly_s + noise + disruptions
        demand = np.clip(demand, 0, None).round().astype(int)

        for i, (date, d) in enumerate(zip(dates, demand)):
            records.append({
                "product_id":   product["id"],
                "product_name": product["name"],
                "category":     product["category"],
                "date":         date.strftime("%Y-%m-%d"),
                "day_of_week":  date.dayofweek,
                "month":        date.month,
                "is_weekend":   int(date.dayofweek >= 5),
                "demand":       d
            })

    df = pd.DataFrame(records)
    log.info(f"Generated {len(df)} demand records")
    return df


def generate_supplier_risk() -> pd.DataFrame:
    log.info("Generating supplier risk data for 100 suppliers...")
    records = []

    for i in range(1, 101):
        supplier_base = SUPPLIERS[i % len(SUPPLIERS)]
        lead_time     = np.random.randint(7, 60)
        lead_var      = np.random.uniform(0, 0.5)
        single_source = np.random.choice([0, 1], p=[0.7, 0.3])
        geo_risk      = round(np.random.uniform(0, 1), 3)
        financial     = round(np.random.uniform(0, 1), 3)
        quality       = round(np.random.uniform(0.6, 1.0), 3)
        on_time       = round(np.random.uniform(0.5, 1.0), 3)
        alternatives  = np.random.randint(0, 5)

        # Risk scoring logic
        risk_score = (
            0.25 * lead_var +
            0.20 * single_source +
            0.20 * geo_risk +
            0.15 * (1 - financial) +
            0.10 * (1 - quality) +
            0.10 * (1 - on_time)
        )

        if risk_score >= 0.5:
            risk_level = "High"
        elif risk_score >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        records.append({
            "supplier_id":            f"SUP-{i:04d}",
            "supplier_name":          f"{supplier_base} — Werk {i}",
            "lead_time_days":         lead_time,
            "lead_time_variance":     round(lead_var, 3),
            "single_source":          single_source,
            "geo_risk_score":         geo_risk,
            "financial_risk_score":   round(1 - financial, 3),
            "quality_score":          quality,
            "on_time_delivery_rate":  on_time,
            "num_alternatives":       alternatives,
            "risk_score":             round(risk_score, 3),
            "risk_level":             risk_level
        })

    df = pd.DataFrame(records)
    log.info(f"Generated {len(df)} supplier records")
    log.info(f"Risk distribution:\n{df['risk_level'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    RISK_DIR.mkdir(parents=True, exist_ok=True)

    demand   = generate_demand_data()
    suppliers = generate_supplier_risk()

    demand.to_csv(DEMAND_DIR    / "demand_data.csv",    index=False)
    suppliers.to_csv(RISK_DIR   / "supplier_risk.csv",  index=False)

    log.info("All data generated successfully.")
    log.info(f"  Demand: {len(demand)} records across {demand['product_id'].nunique()} SKUs")
    log.info(f"  Suppliers: {len(suppliers)} records")