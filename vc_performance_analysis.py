# Datei: vc_performance_analysis.py

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Versuche numpy_financial für XIRR; fallback auf scipy
try:
    import numpy_financial as nf
    has_nf = True
except ImportError:
    from scipy.optimize import newton
    has_nf = False

def load_cashflows(path: str) -> pd.DataFrame:
    """
    Lädt Cashflow-Daten aus einer CSV-Datei.
    Erwartet Spalten:
      - Date (YYYY-MM-DD)
      - Type: 'contribution', 'distribution', 'nav'
      - Amount: Float (negativ für Einzahlungen)
      Optional (identisch in jeder Zeile):
      - FundOpenDate (YYYY-MM-DD)
      - TermYears (Float)
      - TotalCommitment (Float)
    """
    df = pd.read_csv(path, parse_dates=['Date', 'FundOpenDate'], dayfirst=False)
    df['Type'] = df['Type'].str.lower()
    return df

def xnpv(rate: float, cashflows: pd.DataFrame) -> float:
    origin = cashflows['Date'].min()
    return sum(
        cashflows.loc[i, 'Amount'] / (1 + rate) ** ((cashflows.loc[i, 'Date'] - origin).days / 365)
        for i in cashflows.index
    )

def xirr(cashflows: pd.DataFrame) -> float:
    cf = cashflows.copy()
    if has_nf:
        return nf.xirr(cf['Amount'].to_numpy(), cf['Date'].to_numpy())
    else:
        return newton(lambda r: xnpv(r, cf), 0.1)

def compute_metrics(df: pd.DataFrame) -> dict:
    # Basis-Kennzahlen
    contrib = df.loc[df['Type']=='contribution', 'Amount'].sum()
    dist = df.loc[df['Type']=='distribution', 'Amount'].sum()
    nav = df.loc[df['Type']=='nav', 'Amount'].iloc[-1] if 'nav' in df['Type'].values else 0.0

    dpi = dist / contrib if contrib else np.nan
    rvpi = nav / contrib if contrib else np.nan
    tvpi = dpi + rvpi
    moic = tvpi

    # XIRR: NAV als letzte Distribution
    cf = df.copy()
    cf.loc[cf['Type']=='nav', 'Type'] = 'distribution'
    irr = xirr(cf[['Date','Amount']])

    metrics = {
        'Total Contributions': contrib,
        'Total Distributions': dist,
        'NAV': nav,
        'DPI': dpi,
        'RVPI': rvpi,
        'TVPI': tvpi,
        'MOIC': moic,
        'XIRR': irr,
    }

    # Fund Age und FM_real, falls vorhanden
    if 'FundOpenDate' in df.columns and 'TermYears' in df.columns:
        fund_open = df['FundOpenDate'].iloc[0]
        term = float(df['TermYears'].iloc[0])
        last_date = df['Date'].max()
        fund_age = (last_date - fund_open).days / 365.25
        metrics['Fund Age (years)'] = fund_age
        metrics['FM_real'] = min(max(fund_age / term, 0.0), 1.0)

    # Commitment-Kennzahlen aus CSV-Spalte
    if 'TotalCommitment' in df.columns:
        total_comm = float(df['TotalCommitment'].iloc[0])
        metrics['Total Commitment'] = total_comm
        metrics['Remaining Commitment'] = total_comm - metrics['Total Contributions']
        metrics['Drawdown %'] = metrics['Total Contributions'] / total_comm if total_comm else np.nan

    return metrics

# Monte Carlo-Teil (aus Paper) – unverändert
def fm(elapsed: float, term_years: float = 10) -> float:
    return min(max(elapsed / term_years, 0.0), 1.0)

def dpi_model(fm_values: np.ndarray, k: float = 2.0) -> np.ndarray:
    return 1 - np.exp(-k * fm_values)

def rvpi_model(fm_values: np.ndarray, k: float = 2.0) -> np.ndarray:
    return np.exp(-k * fm_values)

def simulate_performance(num_sims: int = 10000, dpi_k: float = 2.0, rvpi_k: float = 2.0, target_tvpi: float = 1.31) -> pd.DataFrame:
    fm_samples = np.random.rand(num_sims)
    dpi = dpi_model(fm_samples, dpi_k)
    rvpi = rvpi_model(fm_samples, rvpi_k)
    tvpi = dpi + rvpi
    scale = target_tvpi / np.median(tvpi)
    return pd.DataFrame({'FM': fm_samples, 'DPI': dpi * scale, 'RVPI': rvpi * scale, 'TVPI': tvpi * scale})

def run_simulation(args):
    import statsmodels.api as sm
    df_sim = simulate_performance(num_sims=args.runs, dpi_k=args.dpi_k, rvpi_k=args.rvpi_k, target_tvpi=args.target_tvpi)
    print(df_sim[['DPI','RVPI','TVPI']].describe())
    model = sm.OLS(df_sim['TVPI'], sm.add_constant(df_sim[['DPI','RVPI']])).fit(cov_type='HC3')
    print(model.summary())

def main():
    parser = argparse.ArgumentParser(description='VC Performance Analyse und Simulation')
    sub = parser.add_subparsers(dest='command', required=True)

    p1 = sub.add_parser('analyze', help='Interim Performance aus Cashflows')
    p1.add_argument('csv_path', help='Pfad zur CSV mit Cashflows + Metadaten')

    p2 = sub.add_parser('simulate', help='Monte Carlo Simulation')
    p2.add_argument('--runs', type=int, default=10000)
    p2.add_argument('--dpi_k', type=float, default=2.0)
    p2.add_argument('--rvpi_k', type=float, default=2.0)
    p2.add_argument('--target_tvpi', type=float, default=1.31)

    args = parser.parse_args()
    if args.command == 'analyze':
        df = load_cashflows(args.csv_path)
        metrics = compute_metrics(df)
        print('--- Analyse-Ergebnisse ---')
        for key, val in metrics.items():
            if isinstance(val, float):
                if key in ['DPI','RVPI','TVPI','MOIC','XIRR','Drawdown %']:
                    print(f"{key}: {val:.2%}")
                else:
                    print(f"{key}: {val:.2f}")
            else:
                print(f"{key}: {val}")
    else:
        run_simulation(args)

if __name__ == '__main__':
    main()

