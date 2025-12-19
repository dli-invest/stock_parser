import requests
import json
import time
import re
import os
import pandas as pd
import numpy as np
from google import genai
from pypdf import PdfReader
from datetime import datetime, timedelta
from pandas import json_normalize, DataFrame # New import for flattening JSON
from gql_queries import GQL

from typing import Union

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK", "YOUR_WEBHOOK_URL")

def configure_genai():
    """Initializes and validates the Gemini configuration."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY":
        raise ValueError("Gemini API Key is missing or not configured.")
    try:
        client = genai.Client()
        # Test model initialization
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to configure Google GenAI: {e}")

def send_to_discord(ticker, analysis_text, filing_url):
    """Sends the formatted analysis to a Discord channel via Webhook."""
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == "YOUR_WEBHOOK_URL":
        print("Warning: Discord Webhook URL not set. Skipping notification.")
        return

    payload = {
        "embeds": [{
            "title": f"🚨 Equity Analysis: {ticker}",
            "description": analysis_text,
            "url": filing_url,
            "color": 3066993,  # Greenish color
            "footer": {"text": "Gemini Financial Intelligence"}
        }]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"Successfully sent {ticker} analysis to Discord.")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def analyze_with_gemini(row, document_text):
    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3-flash-preview", 
        "gemma-3-4b-it"
    ]
    # Prepare the context from the dataframe row
    ticker = row.get('Ticker', 'Unknown')
    mkt_cap = row.get('MarketCap', 'N/A')
    price = row.get('Price', 'N/A')
    # title = row.get('Filing_Title', 'Unknown Filing')
    
    prompt = f"""
    You are a professional Canadian equity analyst. 
    Analyze the following filing for {ticker} (Market Cap: ${mkt_cap}, Price: ${price}).
    
    Document Content (First 5 pages):
    {document_text[:15000]} 
    
    TASK:
    1. Summarize the core news in 2 sentences.
    2. Determine if this is "Market Moving" (Positive, Negative, or Neutral).
    3. Identify any specific financial commitments or changes to share structure.
    4. Provide a 'Sentiment Score' from 1 to 10.
    """
    client = configure_genai()
    try:
        for model_id in models_to_try:
            try:
                print(f"Attempting analysis with {model_id}...")
                
                # The SDK handles basic retries internally, but we catch 429 for model-switching
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2, # Lower temperature for analytical consistency
                    )
                )
                
                # Success! Return the text
                return {
                    "model_used": model_id,
                    "analysis": response.text
                }

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"⚠️ Rate limit hit on {model_id}. Switching to next model...")
                    time.sleep(2) # Brief pause before trying next model
                    continue
                else:
                    return f"Critical Error with {model_id}: {e}"
    except Exception as e:
        return f"Gemini Analysis Error: {e}"

def extract_text_from_pdf(pdf_path):
    """Helper to extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:15]: # Limit to first 5 pages for speed
            text += page.extract_text() + " "
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

def get_ticker_filings(
    symbol: str,
    fromDate: str = datetime.today().replace(day=1).strftime("%Y-%m-%d"),
    toDate: str = datetime.today().strftime("%Y-%m-%d"),
    limit: int = 100,
) -> Union[dict, None]:
    """
    Parameters:
        symbol - ticker symbol from tsx, no prefix
        fromDate - start date to grab documents
        toDate - end date to grab documents
        limit - max number of documents to retrieve
    Returns:
        dict - :ref:`Quote By Symbol <quote_by_symbol_query>`
    """
    payload = GQL.get_company_filings_payload
    payload["variables"]["symbol"] = symbol
    payload["variables"]["fromDate"] = fromDate
    payload["variables"]["toDate"] = toDate
    payload["variables"]["limit"] = limit
    url = "https://app-money.tmx.com/graphql"
    r = requests.post(
        url,
        data=json.dumps(payload),
        headers={
            "authority": "app-money.tmx.com",
            "referer": f"https://money.tmx.com/en/quote/{symbol.upper()}",
            "locale": "en",
            "Content-Type": "application/json",
            # "User-Agent": get_random_agent(),
            "Accept": "*/*"
        },
    )
    try:
        if r.status_code == 403:
            return {}
        else:
            allData = r.json()
            data = allData["data"]
            return data
    except KeyError as _e:
        print(_e, symbol)
        pass

def analyze_recent_filings(df_tickers: DataFrame, days_back: int = 2):
    """
    Iterates through a list of tickers, checks for filings in the last X days,
    downloads them, and extracts text for analysis.
    """
    # 1. Define Date Range (Today and Yesterday)
    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    print(f"\n--- Scanning for filings from {from_date} to {to_date} ---")
    
    os.makedirs("filings_downloads", exist_ok=True)
    report_summary = []

    for index, row in df_tickers.iterrows():
        symbol = row['Ticker']
        print(f"Checking {symbol}...")
        
        # Use your existing GQL function
        filings_data = get_ticker_filings(symbol, fromDate=from_date, toDate=to_date)
        
        if not filings_data or 'filings' not in filings_data:
            continue
            
        filings = filings_data.get('filings', [])
        
        for filing in filings:
            doc_url = filing.get('urlToPdf')
            doc_name = filing.get('description', 'Untitled')
            f_date = filing.get('filingDate')
            
            if not doc_url:
                continue

            print(f"  [!] Found: {doc_name} ({f_date})")

            # if f_date is today then proceed
            
            downloaded_path = download_file(doc_url)
            
            # Extract Text for "Analysis"
            content = ""
            if downloaded_path and downloaded_path.endswith(".pdf"):
                content = extract_text_from_pdf(downloaded_path)
            
            # Placeholder for Analysis Logic (e.g., keyword search or LLM call)
            impact_score = "Manual Review Required"
            if "acquisition" in content.lower() or "merger" in content.lower():
                impact_score = "HIGH - Strategic Event"
            elif "dividend" in content.lower():
                impact_score = "MEDIUM - Capital Distribution"

            report_summary.append({
                "Ticker": symbol,
                "Date": f_date,
                "Document": doc_name,
                "Impact": impact_score,
                "Snippet": content[:200].replace("\n", " ") + "..."
            })

    # Save summary report
    if report_summary:
        summary_df = pd.DataFrame(report_summary)
        summary_df.to_csv("filings_analysis_report.csv", index=False)
        print(f"\n✅ Analysis complete. {len(report_summary)} new filings processed.")
    else:
        print("\nNo new filings found for the selected tickers in the last 48 hours.")

def extract_text_from_pdf(pdf_path):
    """Helper to extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:5]: # Limit to first 5 pages for speed
            text += page.extract_text() + " "
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

def get_ticker_data(symbol=str) -> Union[dict, None]:
    """
    Parameters:
        symbol - ticker symbol from tsx, no prefix
    Returns:
        dict - :ref:`Quote By Symbol <quote_by_symbol_query>`
    """
    payload = GQL.quote_by_symbol_payload
    payload["variables"]["symbol"] = symbol
    url = "https://app-money.tmx.com/graphql"
    r = requests.post(
        url,
        json=payload,
        headers={
            "authority": "app-money.tmx.com",
            "referer": f"https://money.tmx.com/en/quote/{symbol.upper()}",
            "locale": "en",
        },
    )
    try:
        allData = r.json()
    except json.decoder.JSONDecodeError as error:
        print(error)
        print(f"Failed to decode data for {symbol}")
        print(r)
        return None
    # Check for errors
    try:
        data = allData["data"]["getQuoteBySymbol"]
        return data
    except KeyError as _e:
        print(_e, symbol)
        pass


def download_file(url, download_directory='filings_download', output_filename=None):
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        
        # Check if the request was successful (Status Code 200)
        if response.status_code == 200:
            
            # If no filename is provided, try to find it in the Content-Disposition header
            if not output_filename:
                if "Content-Disposition" in response.headers:
                    content_disposition = response.headers["Content-Disposition"]
                    filename_match = re.findall('filename="?([^"]+)"?', content_disposition)
                    if filename_match:
                        output_filename = filename_match[0]
            
            # Fallback filename if one couldn't be determined
            if not output_filename:
                output_filename = "downloaded_file.pdf"

            # Write the content to a file in chunks (good for large files)
            with open(output_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Success: File downloaded as '{output_filename}'")
        else:
            print(f"Error: Failed to download. Status code: {response.status_code}")

        return output_filename
            
    except Exception as e:
        print(f"An error occurred: {e}")

def identify_tickers_with_new_filings(df_tickers: pd.DataFrame) -> pd.DataFrame:
    """
    Scans a list of tickers for filings from 'today'.
    Returns a DataFrame of only those tickers that filed today, including filing metadata.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    print(f"\n--- Identifying tickers with filings for: {today_str} ---")
    
    active_filers = []

    for index, row in df_tickers.iterrows():
        symbol = row['Ticker']
        # We only check for today (fromDate and toDate are today)
        filings_data = get_ticker_filings(symbol, fromDate=today_str, toDate=today_str)
        
        if filings_data and 'filings' in filings_data:
            filings = filings_data.get('filings', [])
            
            if filings:
                print(f"  [+] Found {len(filings)} filing(s) for {symbol}")
                # We grab the most recent filing info to attach to the ticker
                top_filing = filings[0] 
                active_filers.append({
                    "Ticker": symbol,
                    "Filing_Title": top_filing.get('description'),
                    "Filing_URL": top_filing.get('urlToPdf'),
                    "Filing_Date": top_filing.get('filingDate')
                })
        
        # Optional: Add a tiny sleep to avoid rate limiting during the scan
        time.sleep(0.1)

    return pd.DataFrame(active_filers)

def fetch_data_for_active_tickers(df_active: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    """
    Fetches detailed financial data only for the tickers provided in the input DataFrame.
    """
    if df_active.empty:
        return pd.DataFrame()

    print(f"\n--- Fetching financial data for {len(df_active)} active tickers ---")
    detailed_results = []

    for i, symbol in enumerate(df_active['Ticker']):
        print(f"[{i+1}/{len(df_active)}] Fetching financials for: {symbol}")
        
        data = get_ticker_data(symbol)
        if data:
            detailed_results.append(data)
        
        time.sleep(delay)

    if not detailed_results:
        return pd.DataFrame()

    # Flatten the JSON results
    df_financials = json_normalize(detailed_results, sep='_', errors='ignore')
    df_financials["Ticker"] = df_financials['symbol']
    
    # Merge the filing info with the financial data on 'Ticker'
    # We use a right-hand suffix to handle any duplicate columns
    df_final = pd.merge(df_active, df_financials, on='Ticker', how='left')
    
    return df_final


def filter_exchange_listings(file_path: str, ticker_col_name: str, output_file: str) -> pd.DataFrame:
    """
    Loads company listings from a single Excel file, filters out ETFs, Funds,
    and Trusts based on predefined criteria, and saves the resulting tickers.

    Args:
        file_path (str): The path to the Excel file containing the listings.
        ticker_col_name (str): The name of the ticker column in the Excel file.
        output_file (str): The filename to save the resulting non-fund tickers.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered (non-ETF/Fund/Trust) listings.
    """
    
    # 1. Get Sheet Names Dynamically
    try:
        # pd.ExcelFile() is faster than pd.read_excel(sheet_name=None)
        xlsx = pd.ExcelFile(file_path)
        all_sheets = xlsx.sheet_names
        
        # Identify the relevant sheet names for TSX and TSXV data
        # We look for sheets containing 'TSX Issuers' and 'TSXV Issuers'
        tsx_sheet = next((s for s in all_sheets if 'TSX Issuers' in s), None)
        tsxv_sheet = next((s for s in all_sheets if 'TSXV Issuers' in s), None)

        if not tsx_sheet or not tsxv_sheet:
            print("Error: Could not find both 'TSX Issuers' and 'TSXV Issuers' sheets.")
            return pd.DataFrame() # Return empty DataFrame on failure

        print(f"✅ Found TSX Sheet: '{tsx_sheet}'")
        print(f"✅ Found TSXV Sheet: '{tsxv_sheet}'")

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the Excel file structure: {e}")
        return pd.DataFrame()

    # 2. Load and Prepare DataFrames
    header_row = 9 # Assuming the data starts after 9 header rows (10th row)
    
    # --- Load TSX data ---
    df_tsx = pd.read_excel(file_path, header=header_row, sheet_name=tsx_sheet)
    # Ensure required columns exist for consistent concatenation/filtering
    if 'Trust' not in df_tsx.columns:
        df_tsx['Trust'] = np.nan
    if 'Sector' not in df_tsx.columns:
        df_tsx['Sector'] = np.nan
    df_tsx = df_tsx.rename(columns={ticker_col_name: 'Ticker'})


    # --- Load TSXV data ---
    df_tsxv = pd.read_excel(file_path, header=header_row, sheet_name=tsxv_sheet)
    # Ensure required columns exist for consistent concatenation/filtering
    if 'SP_Type' not in df_tsxv.columns:
        df_tsxv['SP_Type'] = np.nan
    if 'Sector' not in df_tsxv.columns:
        df_tsxv['Sector'] = np.nan
    
    df_tsxv = df_tsxv.rename(columns={ticker_col_name: 'Ticker'})

    # elif 'Ticker' not in df_tsxv.columns:
    #      # Fallback error check if neither name is present after loading
    #      print("Error: 'Ticker' column not found in TSXV data after processing.")
    #      return pd.DataFrame()


    # 3. Concatenate and Standardize Ticker Column
    # Using 'Ticker' as the consistent name after individual renames
    df_combined = pd.concat([df_tsx, df_tsxv], ignore_index=True)


    # 4. Define ETF/Fund/Trust Exclusion Criteria (Your original logic)
    # Note: Using .str.strip().str.upper() on string columns to catch minor inconsistencies
    
    # Criteria 1: Exclude common fund sectors (ETP - Exchange Traded Product, Closed-End Funds)
    fund_sectors = ['ETP', 'CLOSED-END FUNDS']
    df_combined['Sector_Clean'] = df_combined['Sector'].astype(str).str.strip().str.upper()
    is_fund_sector = df_combined['Sector_Clean'].isin(fund_sectors)

    # Criteria 2: Exclude specific fund types from TSX (Exchange Traded Funds and other fund structures)
    fund_sp_types = ['EXCHANGE TRADED FUNDS', 'INCOME TRUST', 'CDR', 'SPLIT SHARES', 'FUND OF EQUITIES', 
                     'COMMODITY FUNDS', 'FUND OF DEBT', 'FUND OF MULTI-ASSET/OTHER', 'EXCHANGE TRADED RECEIPT']
    df_combined['SP_Type_Clean'] = df_combined['SP_Type'].astype(str).str.strip().str.upper()
    is_fund_sp_type = df_combined['SP_Type_Clean'].isin(fund_sp_types)

    # Criteria 3: Exclude rows explicitly marked as a Trust from TSXV
    is_trust = (df_combined['Trust'] == 'Y') | (df_combined['Trust'] == 'y') # Handle case sensitivity

    # --- Filter out ETFs/Funds/Trusts ---
    # Create a mask that is TRUE for rows to be EXCLUDED
    is_etf_or_fund = is_fund_sector | is_fund_sp_type | is_trust

    # Filter the DataFrame to keep only non-ETF/Fund/Trusts
    df_filtered = df_combined[~is_etf_or_fund].copy()
    
    # 5. Output the results
    df_filtered.to_csv(output_file, index=False)

    # Print summary
    print("--- Results Summary ---")
    print(f"Total initial combined listings: {len(df_combined)}")
    print(f"Total unique tickers EXCLUDED: {is_etf_or_fund.sum()}")
    print(f"Total unique tickers (non-ETF/Fund/Trust): {len(df_filtered)}")
    print(f"Full list of tickers saved to {output_file}")
    
    return df_filtered

def filter_tickers(file_path: str, filter_list: list, final_output_file: str) -> pd.DataFrame:
    """
    Loads the full listings CSV, applies dynamic filters, and saves the final result.

    The filter_list format now supports both numeric and categorical (industry) filters:
    - Numeric: [{"type": "numeric", "column_key": "Market Cap", "operator": ">=", "threshold": 4e7}]
    - Categorical: [{"type": "categorical", "column_key": "Sector", "operation": "exclude", "values": ["MINING", "OIL & GAS"]}]

    Args:
        file_path (str): Path to the CSV file (output from filter_exchange_listings).
        filter_list (list): List of dictionaries, where each dict contains the filter.
        final_output_file (str): The filename to save the final filtered listings.

    Returns:
        pd.DataFrame: The final filtered DataFrame.
    """
    print(f"\n--- Applying Additional Filters from {file_path} ---")
    try:
        # We read the full data CSV that contains all columns needed for filtering (e.g., 'Sector')
        df = pd.read_csv(file_path) 
    except FileNotFoundError:
        print(f"Error: Listings file not found at path: {file_path}")
        return pd.DataFrame()

    df_current = df.copy()
    available_columns = df.columns.tolist()
    print(f"Columns available for filtering: {available_columns}")

    for filter_dict in filter_list:
        try:
            filter_type = filter_dict['type']
            filter_key = filter_dict['column_key']
        except KeyError:
            print(f"Warning: Filter dictionary is missing required keys ('type' or 'column_key'). Skipping: {filter_dict}")
            continue

        # --- Dynamic Column Name Resolution ---
        actual_column = next((col for col in available_columns if col.startswith(filter_key)), None)

        if actual_column is None:
            print(f"Warning: No column found starting with the key '{filter_key}'. Skipping this filter.")
            continue
        
        print(f"Mapping filter key '{filter_key}' to actual column: '{actual_column}'")
        
        original_length = len(df_current)

        if filter_type == 'numeric':
            # --- NUMERIC FILTER LOGIC (Existing Logic) ---
            try:
                operator = filter_dict['operator']
                threshold = filter_dict['threshold']
            except KeyError:
                print(f"Warning: Numeric filter is missing required keys ('operator' or 'threshold'). Skipping: {filter_dict}")
                continue
            
            print(f"Applying NUMERIC filter: '{actual_column}' {operator} {threshold:,.0f}")

            # 1. Ensure the column is numeric and coerce errors to NaN
            df_current[actual_column] = pd.to_numeric(df_current[actual_column], errors='coerce')

            # 2. Apply the dynamic filter
            if operator == '>':
                df_current = df_current[df_current[actual_column] > threshold].copy()
            elif operator == '>=':
                df_current = df_current[df_current[actual_column] >= threshold].copy()
            elif operator == '<':
                df_current = df_current[df_current[actual_column] < threshold].copy()
            elif operator == '<=':
                df_current = df_current[df_current[actual_column] <= threshold].copy()
            elif operator == '==':
                df_current = df_current[df_current[actual_column] == threshold].copy()
            else:
                print(f"Warning: Invalid numeric operator '{operator}'. Skipping this filter.")
                continue

        elif filter_type == 'categorical':
            # --- NEW CATEGORICAL FILTER LOGIC ---
            try:
                operation = filter_dict['operation'] # 'include' or 'exclude'
                values = [str(v).upper() for v in filter_dict['values']] # List of industry/sector names
            except KeyError:
                print(f"Warning: Categorical filter is missing required keys ('operation' or 'values'). Skipping: {filter_dict}")
                continue

            print(f"Applying CATEGORICAL filter: '{actual_column}' {operation} list {values}")
            
            # 1. Standardize the column values for case-insensitive comparison
            # Coerce to string first, then strip, then uppercase
            df_current['temp_clean_col'] = df_current[actual_column].astype(str).str.strip().str.upper()
            
            if operation == 'include':
                # Keep rows where the value is IN the list
                df_current = df_current[df_current['temp_clean_col'].isin(values)].copy()
            elif operation == 'exclude':
                # Keep rows where the value is NOT IN the list
                df_current = df_current[~df_current['temp_clean_col'].isin(values)].copy()
            else:
                print(f"Warning: Invalid categorical operation '{operation}'. Must be 'include' or 'exclude'. Skipping.")
                df_current = df_current.drop(columns=['temp_clean_col'])
                continue

            # 2. Clean up the temporary column used for filtering
            df_current = df_current.drop(columns=['temp_clean_col'])


        else:
            print(f"Warning: Invalid filter type '{filter_type}'. Must be 'numeric' or 'categorical'. Skipping.")
            continue
            
        print(f"  Filtered from {original_length} to {len(df_current)} entries.")

    # Select only the Ticker column for the final output
    df_current.to_csv(final_output_file, index=False)

    print(f"\n--- Final Filter Summary ---")
    print(f"Total entries after all filters: {len(df_current)}")
    print(f"Final list of listings saved to {final_output_file}")

    return df_current

def fetch_all_ticker_data(ticker_list_file: str, delay_seconds: float, final_data_output: str) -> pd.DataFrame:
    """
    Reads a list of tickers, fetches detailed data for each, and saves the complete
    result as a flattened DataFrame with each key from the JSON as its own column.

    Args:
        ticker_list_file (str): Path to the CSV containing the 'Ticker' column.
        delay_seconds (float): Time to pause between each API call in seconds.
        final_data_output (str): Filename to save the final flattened DataFrame.

    Returns:
        pd.DataFrame: A flattened DataFrame with one row per ticker and all 
                      JSON keys mapped to columns.
    """
    print(f"\n--- Starting Data Fetch for Tickers from {ticker_list_file} ---")
    
    try:
        df_tickers = pd.read_csv(ticker_list_file)
        if 'Ticker' not in df_tickers.columns:
            print("Error: The input file must contain a column named 'Ticker'.")
            return pd.DataFrame()
        
        tickers = df_tickers['Ticker'].astype(str).tolist()
        tickers = list(set(tickers)) 
        print(f"Total unique tickers to process: {len(tickers)}")

    except FileNotFoundError:
        print(f"Error: Ticker list file not found at path: {ticker_list_file}")
        return pd.DataFrame()

    # List to store the full dictionary results (no string conversion yet)
    successful_fetches = []
    COMMON_EXTENSIONS = ['.PR.A', '.A',  '.CNX', '.V', '.CN', '.UN']
    # Iterate through the tickers and fetch data
    for i, symbol in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Fetching data for: {symbol}")
        
        data = get_ticker_data(symbol)
        found_data = (data is not None)
        fetch_symbol = symbol

        # --- Retry Logic with Extensions ---
        if not found_data:
            print(f"Retrying {symbol} due to fetch error.")
            
            for ext in COMMON_EXTENSIONS:
                retry_symbol = symbol + ext
                print(f"  -> Trying extended symbol: {retry_symbol}")
                data = get_ticker_data(retry_symbol)
                
                if data is not None:
                    found_data = True
                    fetch_symbol = retry_symbol
                    print(f"  -> SUCCESS with {fetch_symbol}")
                    break # Stop looping extensions once data is found
            
            if not found_data:
                 print(f"Skipping {symbol}: Failed after all retries.")
                 continue
        if found_data:
            successful_fetches.append(data)
        # Apply the sequential delay (best practice for APIs)
        if i < len(tickers) - 1:
            time.sleep(delay_seconds)

    print("\n--- Data Flattening and Processing ---")
    if not successful_fetches:
        print("No successful data fetches. Returning empty DataFrame.")
        return pd.DataFrame()
        
    # 1. Use json_normalize to flatten the nested list of dictionaries
    # 'Ticker' is excluded from being treated as nested data.
    # We use a separator (e.g., '_') for keys in nested dictionaries (e.g., 'price_last').
    df_final = json_normalize(
        successful_fetches, 
        sep='_', 
        errors='ignore' # Handles dictionaries with different structures
    )
    
    # Optional: Reorder columns to put 'Ticker' first
    cols = ['Ticker'] + [col for col in df_final.columns if col != 'Ticker']
    df_final = df_final[cols]


    # 2. Save and log the final result
    df_final.to_csv(final_data_output, index=False)
    
    print(f"Total tickers processed successfully: {len(df_final)}")
    print(f"Final flattened data saved to '{final_data_output}'")
    print(f"Resulting DataFrame has {len(df_final.columns)} columns.")
    print("\n--- First 5 Rows (Transposed for Preview) ---")
    # Display the first few rows, transposed, to show the new column structure
    print(df_final.head().T) 

    return df_final

if __name__ == "__main__":
    url = "https://www.tsx.com/en/resource/571"
    file_tsx_tsxv = download_file(url)
    # file_tsx_tsxv ='tsx-and-amp-tsxv-listed-companies-2025-12-15-en.xlsx'
    ticker_col_name_actual = 'Root\nTicker' 
    output_csv_file = 'non_etf_tickers.csv'

    # # Execute the function
    df_result = filter_exchange_listings(
        file_path=file_tsx_tsxv, 
        ticker_col_name=ticker_col_name_actual, 
        output_file=output_csv_file
    )
    new_filter_list = [
        # Numeric Filter: Market Cap >= 30 million
        {"type": "numeric", "column_key": "Market Cap", "operator": ">=", "threshold": 1e7},
        
        # Numeric Filter: Market Cap < 300 million
        {"type": "numeric", "column_key": "Market Cap", "operator": "<", "threshold": 1e9}, 
        
        # Categorical Filter: EXCLUDE companies in Mining and Oil & Gas sectors
        # {"type": "categorical", 
        #  "column_key": "Sector", 
        #  "operation": "exclude", 
        #  "values": ["MINING"]
        # },
        
        # Categorical Filter (Optional Example): INCLUDE only Technology and Financial Services sectors
        # This will filter the results *after* the Market Cap and EXCLUDE filters
        # {"type": "categorical",
        #  "column_key": "Sector",
        #  "operation": "include",
        #  "values": ["TECHNOLOGY", "FINANCIAL SERVICES"]
        # }
    ]
    df_filtered = filter_tickers(
        file_path=output_csv_file, 
        filter_list=new_filter_list, 
        final_output_file='filtered.csv'
    )

    df_active_filers = identify_tickers_with_new_filings(df_filtered)

    if not df_active_filers.empty:
        # 4. FETCH detailed financial data for ONLY those active tickers
        df_final_report = fetch_data_for_active_tickers(df_active_filers, delay=0.5)
        
        # 5. Output Final Results
        df_final_report.to_csv("active_filings_with_financials.csv", index=False)
        
        print("\n--- Final Report Ready ---")
        print(f"Processed {len(df_final_report)} tickers with new filings.")
        print(df_final_report[['Ticker', 'Filing_Title', 'Filing_URL']].head())
        # entries
        # entries = df_final_report.head()
        # for idx, row in df_final_report.iterrows():
        for idx, row in df_final_report.iterrows():
            print("using gemini to scan through entries")
            ticker = row['Ticker']
            filing_url = row['Filing_URL']
            # download the file in the list, extract the text, print that the text is extracted
            file_path = download_file(filing_url)
            if file_path and os.path.exists(file_path):
                content = extract_text_from_pdf(file_path)
                analysis_results = analyze_with_gemini(row, content)
                # check if analysis_results is a dict, if so extract
                if analysis_results.get('analysis'):
                    analysis_results = analysis_results
                send_to_discord(ticker, analysis_results, filing_url)
            else:
                print(f"Skipping {ticker} due to download failure.")
                send_to_discord(ticker, f"Skipping {ticker} due to download failure.", filing_url)

            # assuming 20 rpm, so wait 5 seconds per entry
            time.sleep(10)
    else:
        print("\nNo companies in your filtered universe filed documents today.")

    # API_DELAY_SECONDS = 0.5  
    # final_output_all_data = 'final_ticker_data_flattened.csv'

    # df_final_data = fetch_all_ticker_data(
    #     ticker_list_file="filtered.csv", 
    #     delay_seconds=API_DELAY_SECONDS, 
    #     final_data_output=final_output_all_data
    # )
    # df_final_data = pd.read_csv(final_output_all_data)
    # if not df_final_data.empty:
    #     # 2. Run the Filings Analysis
    #     analyze_recent_filings(df_final_data, days_back=0)
    # else:
    #     print("No tickers passed the filters; skipping filings analysis.")

    # iterate through all the tickers, check for news in the past 3 days

    # print all the filings
    