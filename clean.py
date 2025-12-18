import pandas as pd
import numpy as np

# File names
file_tsx = 'tsx-and-amp-tsxv-listed-companies-2025-12-15-en.xlsx'
file_tsxv = 'tsx-and-amp-tsxv-listed-companies-2025-12-15-en.xlsx'
ticker_col_name = 'Root\nTicker' # The actual column name in the file

# TODO dynamically pull name based on data
# --- Load and prepare TSX data ---
# --- Load and prepare TSX data ---
df_tsx = pd.read_excel(file_tsx, header=9, sheet_name='TSX Issuers November 2025')
# Add 'Trust' column for consistent concatenation with TSXV
if 'Trust' not in df_tsx.columns:
    df_tsx['Trust'] = np.nan

if 'Sector' not in df_tsx.columns:
    df_tsx['Sector'] = np.nan

df_tsx = df_tsx.rename(columns={ticker_col_name: 'Ticker'})


# --- Load and prepare TSXV data ---
df_tsxv = pd.read_excel(file_tsxv, header=9, sheet_name='TSXV Issuers November 2025')
# Add 'SP_Type' column for consistent concatenation with TSX
if 'SP_Type' not in df_tsxv.columns:
    df_tsxv['SP_Type'] = np.nan

if 'Sector' not in df_tsxv.columns:
    df_tsxv['Sector'] = np.nan


# --- Concatenate DataFrames (retaining all columns) ---
df_combined = pd.concat([df_tsx, df_tsxv], ignore_index=True)

df_combined = df_combined.rename(columns={ticker_col_name: 'Ticker'})

# --- Define ETF/Fund/Trust Exclusion Criteria ---
# Criteria 1: Exclude common fund sectors (ETP - Exchange Traded Product, Closed-End Funds)
fund_sectors = ['ETP', 'Closed-End Funds']
is_fund_sector = df_combined['Sector'].isin(fund_sectors)

# Criteria 2: Exclude specific fund types from TSX (Exchange Traded Funds and other fund structures)
fund_sp_types = ['Exchange Traded Funds', 'Income Trust', 'CDR', 'Split Shares', 'Fund of Equities', 'Commodity Funds', 'Fund of Debt', 'Fund of Multi-Asset/Other', 'Exchange Traded Receipt']
is_fund_sp_type = df_combined['SP_Type'].isin(fund_sp_types)

# Criteria 3: Exclude rows explicitly marked as a Trust from TSXV
is_trust = (df_combined['Trust'] == 'Y')

# --- Filter out ETFs/Funds/Trusts ---
# Create a mask that is TRUE for rows to be EXCLUDED
is_etf_or_fund = is_fund_sector | is_fund_sp_type | is_trust

# Filter the DataFrame to keep only non-ETF/Fund/Trusts
df_filtered = df_combined[~is_etf_or_fund].copy()


# --- Output the results ---
# Write the full list to a CSV file
df_tickers = pd.DataFrame(df_filtered)
output_file = 'non_etf_tickers.csv'
df_tickers.to_csv(output_file, index=False)

print(f"Total unique tickers (non-ETF/Fund/Trust): {len(df_tickers)}")
print(f"Full list of tickers saved to {output_file}")