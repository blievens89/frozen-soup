import os
import io
import time
import json
import re
from datetime import datetime
from collections import defaultdict
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from http.client import HTTPSConnection
from base64 import b64encode
from json import loads
from json import dumps

# RestClient class
class RestClient:
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def request(self, path, method, data=None, max_retries=3):
        """Make API request with retry logic"""
        for attempt in range(max_retries):
            connection = HTTPSConnection(self.domain)
            try:
                base64_bytes = b64encode(
                    ("%s:%s" % (self.username, self.password)).encode("ascii")
                ).decode("ascii")
                headers = {'Authorization': 'Basic %s' % base64_bytes, 'Content-Encoding': 'gzip'}

                if data:
                    data_str = dumps(list(data.values()))
                else:
                    data_str = None

                connection.request(method, path, headers=headers, body=data_str)
                response = connection.getresponse()
                return loads(response.read().decode())
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
            finally:
                connection.close()

    def get(self, path):
        return self.request(path, 'GET')

    def post(self, path, data):
        return self.request(path, 'POST', data)

# ============================================
# HELPER FUNCTIONS FOR ANALYSIS
# ============================================

def calculate_keyword_difficulty(competition, cpc):
    """Calculate keyword difficulty score (0-100)"""
    # Higher competition and CPC = higher difficulty
    comp_score = competition * 50  # 0-1 scale to 0-50
    cpc_normalized = min(cpc / 10, 1) * 50  # Normalize CPC, cap at £10 for max score
    return min(comp_score + cpc_normalized, 100)

def calculate_opportunity_score(search_volume, cpc, competition, intent):
    """Calculate opportunity score (0-100) - higher is better"""
    # High volume, reasonable CPC, lower competition = high opportunity
    intent_multiplier = {
        'transactional': 1.2,
        'commercial': 1.1,
        'navigational': 0.9,
        'informational': 0.8,
        'unknown': 0.7
    }.get(intent, 0.7)

    # Normalize search volume (log scale)
    volume_score = min(np.log10(search_volume + 1) / 5, 1) * 40

    # CPC sweet spot: £0.50-£3.00 is ideal
    if 0.5 <= cpc <= 3.0:
        cpc_score = 30
    elif cpc < 0.5:
        cpc_score = 15 * (cpc / 0.5)  # Lower is less valuable
    else:
        cpc_score = 30 * (1 / (1 + (cpc - 3.0) / 3))  # Higher diminishes value

    # Lower competition is better
    competition_score = (1 - competition) * 30

    raw_score = (volume_score + cpc_score + competition_score) * intent_multiplier
    return min(raw_score, 100)

def group_keywords_by_similarity(keywords_df):
    """Group keywords by common themes/words"""
    groups = defaultdict(list)

    for idx, row in keywords_df.iterrows():
        keyword = row['keyword'].lower()
        words = set(re.findall(r'\b\w+\b', keyword))

        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = words - stop_words

        # Find best matching group or create new one
        best_group = None
        best_overlap = 0

        for group_key, group_keywords in groups.items():
            group_words = set(re.findall(r'\b\w+\b', group_key))
            overlap = len(words & group_words)
            if overlap > best_overlap and overlap >= len(words) * 0.5:
                best_overlap = overlap
                best_group = group_key

        if best_group:
            groups[best_group].append(keyword)
        else:
            # Use the most common word as group key
            if words:
                key = sorted(words, key=len, reverse=True)[0]
                groups[key].append(keyword)

    # Assign group names to dataframe
    keyword_to_group = {}
    for group_name, kw_list in groups.items():
        for kw in kw_list:
            keyword_to_group[kw] = group_name.title()

    keywords_df['keyword_group'] = keywords_df['keyword'].str.lower().map(keyword_to_group).fillna('Other')
    return keywords_df

def suggest_negative_keywords(keywords_df):
    """Suggest potential negative keywords based on low intent/high cost/low volume"""
    negatives = []

    for idx, row in keywords_df.iterrows():
        keyword = row['keyword']
        reasons = []

        # Low search volume
        if row.get('search_volume', 0) < 10:
            reasons.append('very low search volume (<10)')

        # Informational with high CPC
        if row.get('intent') == 'informational' and row.get('cpc_gbp', 0) > 1.5:
            reasons.append('informational intent with high CPC')

        # Very high competition with low volume
        if row.get('competition', 0) > 0.8 and row.get('search_volume', 0) < 50:
            reasons.append('high competition, low volume')

        # Check for common negative indicators
        negative_indicators = ['free', 'cheap', 'job', 'jobs', 'career', 'salary', 'course', 'training', 'diy', 'how to make']
        keyword_lower = keyword.lower()
        for indicator in negative_indicators:
            if indicator in keyword_lower and row.get('intent') != 'transactional':
                reasons.append(f'contains "{indicator}"')
                break

        if reasons:
            negatives.append({
                'keyword': keyword,
                'reason': '; '.join(reasons),
                'search_volume': row.get('search_volume', 0),
                'cpc_gbp': row.get('cpc_gbp', 0),
                'intent': row.get('intent', 'unknown')
            })

    return pd.DataFrame(negatives) if negatives else pd.DataFrame()

def calculate_roi_metrics(df, avg_order_value, profit_margin):
    """Calculate ROI metrics with revenue projections"""
    df = df.copy()

    # Revenue per conversion
    df['revenue_per_conversion'] = avg_order_value
    df['profit_per_conversion'] = avg_order_value * profit_margin

    # Total revenue and profit
    df['total_revenue'] = df['Conversions'] * df['revenue_per_conversion']
    df['total_profit'] = df['Conversions'] * df['profit_per_conversion']

    # ROI and ROAS
    df['ROI'] = np.where(
        df['Spend £'] > 0,
        ((df['total_profit'] - df['Spend £']) / df['Spend £'] * 100).round(2),
        0
    )
    df['ROAS'] = np.where(
        df['Spend £'] > 0,
        (df['total_revenue'] / df['Spend £']).round(2),
        0
    )

    return df

def optimize_budget_allocation(summary_df, total_budget):
    """Optimize budget allocation based on expected ROAS/CPA"""
    # Use CPA as inverse weight - lower CPA gets more budget
    summary_df = summary_df.copy()

    # Calculate efficiency score (inverse of CPA, normalized)
    min_cpa = summary_df['CPA £'].replace(0, np.inf).min()
    summary_df['efficiency'] = np.where(
        summary_df['CPA £'] > 0,
        min_cpa / summary_df['CPA £'],
        0
    )

    total_efficiency = summary_df['efficiency'].sum()

    if total_efficiency > 0:
        summary_df['optimized_budget_pct'] = (summary_df['efficiency'] / total_efficiency * 100).round(1)
        summary_df['optimized_budget'] = (summary_df['optimized_budget_pct'] * total_budget / 100).round(2)
    else:
        # Equal distribution if we can't calculate efficiency
        n_intents = len(summary_df)
        summary_df['optimized_budget_pct'] = 100 / n_intents
        summary_df['optimized_budget'] = total_budget / n_intents

    return summary_df

def parse_negative_keywords(input_text):
    """Parse negative keywords from text input (comma or newline separated)"""
    if not input_text or not input_text.strip():
        return []

    # Split by newlines and commas
    negatives = []
    for line in input_text.split('\n'):
        for term in line.split(','):
            term = term.strip().lower()
            if term:
                negatives.append(term)

    return list(set(negatives))  # Remove duplicates

def apply_negative_keywords(df, negative_terms, match_type="Contains"):
    """Filter out keywords based on negative terms"""
    if not negative_terms or df.empty:
        return df, 0

    original_count = len(df)
    df_filtered = df.copy()

    if match_type == "Exact Match":
        # Exact match: keyword must exactly match one of the negative terms
        df_filtered = df_filtered[~df_filtered['keyword'].str.lower().isin(negative_terms)]
    else:
        # Contains: keyword must not contain any of the negative terms
        for term in negative_terms:
            df_filtered = df_filtered[~df_filtered['keyword'].str.lower().str.contains(term, regex=False, na=False)]

    filtered_count = original_count - len(df_filtered)
    return df_filtered, filtered_count

# Streamlit configuration
st.set_page_config(page_title="Labs Keyword Ideas + Intent (Live)", layout="wide")
st.title("DataForSEO Labs — Keyword & Intent Planner")

if 'results' not in st.session_state:
    st.session_state.results = None

if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    password_input = st.text_input("Password", type="password")
    if st.button("Enter"):
        if password_input == st.secrets.get("APP_PASSWORD"):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("The password you entered is incorrect.")
    st.stop()

# API credentials
DATAFORSEO_LOGIN = st.secrets.get("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = st.secrets.get("DATAFORSEO_PASSWORD")

if not (DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD):
    st.error("DataForSEO API credentials are not found. Please set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in your Streamlit secrets.")
    st.stop()

try:
    client = RestClient(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD)
    BASE_URL = "/v3"
except Exception as e:
    st.error(f"Failed to initialise the API client: {e}")
    st.stop()

def make_api_post_request(endpoint: str, payload: dict) -> dict:
    try:
        response = client.post(f"{BASE_URL}{endpoint}", payload)
        if response and response.get("status_code") == 20000:
            return response
        else:
            st.error(f"API Error on {endpoint}: {response.get('status_message', 'Unknown error')}")
            if st.session_state.get('show_raw_data', False):
                st.json(response)
            return {}
    except Exception as e:
        st.error(f"An exception occurred while calling the API endpoint {endpoint}: {e}")
        return {}

def extract_items_from_response(response: dict) -> list:
    try:
        if not response or response.get("tasks_error", 1) > 0:
            st.warning("The API task returned an error. See raw response for details.")
            return []
        
        tasks = response.get("tasks", [])
        if not tasks:
            return []
            
        result = tasks[0].get("result")
        if not result:
            return []
            
        return result
    except (KeyError, IndexError, TypeError) as e:
        st.error(f"Error extracting items from response: {e}")
        return []

def safe_average(series: pd.Series) -> float:
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    return numeric_series.mean() if not numeric_series.empty else 0.0

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    
    analysis_mode = st.radio(
        "Analysis Mode",
        ("Generate from Seed Keyword", "Analyse My Keyword List", "Scan Website for Keywords"),
        key="analysis_mode"
    )

    language_code = st.text_input("Language Code (e.g., en, fr, de)", value="en")
    location_name = st.text_input("Location Name", value="United Kingdom")
    
    if analysis_mode == "Generate from Seed Keyword":
        seed_keyword = st.text_input("Seed Keyword", value="remortgage")
        limit = st.slider("Max Keyword Ideas", 10, 300, 50, step=10)
        uploaded_keywords = None
        target_url = None
    elif analysis_mode == "Analyse My Keyword List":
        st.subheader("Your Keywords")
        pasted_keywords = st.text_area("Paste keywords here (one per line)")
        uploaded_file = st.file_uploader("Or upload a TXT/CSV file", type=['txt', 'csv'])
        
        uploaded_keywords = []
        if pasted_keywords:
            lines = pasted_keywords.splitlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            uploaded_keywords.extend(cleaned_lines)
        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file, header=None)
                lines = df_upload[0].dropna().astype(str).tolist()
                uploaded_keywords.extend(lines)
            except Exception as e:
                st.error(f"Error reading file: {e}")

        uploaded_keywords = list(dict.fromkeys(filter(None, uploaded_keywords)))
        seed_keyword = None
        target_url = None
    else: # Scan Website
        target_url = st.text_input("Enter URL to scan", value="https://www.gov.uk/remortgaging-your-home")
        limit = st.slider("Max Keyword Ideas", 10, 300, 50, step=10)
        uploaded_keywords = None
        seed_keyword = None

    usd_to_gbp_rate = st.number_input("USD to GBP Exchange Rate", 0.1, 2.0, 0.79, 0.01)

    st.divider()
    st.header("Pre-Filter Negative Keywords")
    st.caption("Exclude keywords containing these terms (applied before analysis)")
    pre_negatives_input = st.text_area(
        "Enter negative keywords (one per line or comma-separated)",
        value="",
        height=100,
        placeholder="e.g., free, cheap, job, jobs\ncareer, salary",
        key="pre_negatives"
    )

    match_type = st.radio(
        "Match Type",
        ("Contains", "Exact Match"),
        help="Contains: Filter if keyword contains the term. Exact: Filter only if exact match."
    )

    # Show active negative keywords count
    if pre_negatives_input.strip():
        active_negatives = parse_negative_keywords(pre_negatives_input)
        if active_negatives:
            st.info(f"✓ {len(active_negatives)} negative term(s) active: {', '.join(active_negatives[:5])}{'...' if len(active_negatives) > 5 else ''}")

    st.divider()
    st.caption("CTR/CVR Assumptions by Intent")
    intents = ["informational", "navigational", "commercial", "transactional"]
    ctr_defaults = {"informational": 0.03, "navigational": 0.03, "commercial": 0.04, "transactional": 0.04}
    cvr_defaults = {"informational": 0.015, "navigational": 0.015, "commercial": 0.03, "transactional": 0.03}
    ctrs, cvrs = {}, {}
    for intent in intents:
        col1, col2 = st.columns(2)
        with col1:
            ctrs[intent] = st.number_input(f"{intent.title()} CTR", 0.0, 1.0, ctr_defaults[intent], 0.005, format="%.3f", key=f"ctr_{intent}")
        with col2:
            cvrs[intent] = st.number_input(f"{intent.title()} CVR", 0.0, 1.0, cvr_defaults[intent], 0.005, format="%.3f", key=f"cvr_{intent}")

    st.divider()
    st.header("Advanced Filters")
    enable_filters = st.toggle("Enable Filtering", value=False)

    if enable_filters:
        min_search_volume = st.number_input("Min Search Volume", min_value=0, value=10, step=10)
        max_search_volume = st.number_input("Max Search Volume (0 = no limit)", min_value=0, value=0, step=100)

        min_cpc = st.number_input("Min CPC (£)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        max_cpc = st.number_input("Max CPC (£, 0 = no limit)", min_value=0.0, value=0.0, step=0.5, format="%.2f")

        min_competition = st.slider("Min Competition", 0.0, 1.0, 0.0, 0.1)
        max_competition = st.slider("Max Competition", 0.0, 1.0, 1.0, 0.1)

        intent_filter = st.multiselect("Filter by Intent", intents + ['unknown'], default=[])
    else:
        min_search_volume = 0
        max_search_volume = 0
        min_cpc = 0.0
        max_cpc = 0.0
        min_competition = 0.0
        max_competition = 1.0
        intent_filter = []

    st.divider()
    st.header("Budgeting & ROI")
    use_custom_budget = st.toggle("Set Custom Budget", value=False)
    if use_custom_budget:
        total_budget = st.number_input("Total Monthly Budget (£)", min_value=0.0, value=1000.0, step=100.0)
        st.caption("Allocate budget by intent (%)")

        budget_allocations = {}
        for intent in intents:
            budget_allocations[intent] = st.number_input(f"{intent.title()} %", min_value=0, max_value=100, value=25, key=f"budget_{intent}")

    enable_roi_calc = st.toggle("Enable ROI Calculator", value=False)
    if enable_roi_calc:
        avg_order_value = st.number_input("Average Order Value (£)", min_value=0.0, value=100.0, step=10.0)
        profit_margin = st.slider("Profit Margin (%)", 0, 100, 30, 5) / 100

    show_budget_optimization = st.toggle("Show Budget Optimization", value=False)

    st.divider()
    show_raw_data = st.toggle("Show Raw API Data (for debugging)", value=False)
    st.session_state.show_raw_data = show_raw_data

@st.cache_data(ttl=3600, show_spinner="Fetching keyword suggestions...")
def get_keyword_suggestions(seed: str, lang_code: str, loc_name: str, limit: int) -> pd.DataFrame:
    payload_item = {
        "keyword": seed.strip(),
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
        "limit": limit,
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/keyword_suggestions/live", post_data)
    
    if st.session_state.get('show_raw_data', False):
        st.json(response)
    
    items = extract_items_from_response(response)
    
    if not items:
        st.warning("No data returned from keyword suggestions API")
        return pd.DataFrame()

    rows = []
    try:
        # Debug: Check what we actually got
        if st.session_state.get('show_raw_data', False):
            st.write("Items structure:", type(items))
            st.write("Items content:", items)
        
        # Handle different possible response structures
        items_data = None
        
        if isinstance(items, list):
            if len(items) > 0:
                first_item = items[0]
                if isinstance(first_item, dict):
                    if 'items' in first_item:
                        items_data = first_item['items']
                    elif 'keyword' in first_item:  # Direct keyword data
                        items_data = items
                    else:
                        # Look for other possible data containers
                        for key in ['data', 'results', 'keywords']:
                            if key in first_item:
                                items_data = first_item[key]
                                break
                else:
                    items_data = items
            else:
                items_data = []
        elif isinstance(items, dict):
            # If items is a dict, look for keyword data
            if 'items' in items:
                items_data = items['items']
            elif 'data' in items:
                items_data = items['data']
            elif 'results' in items:
                items_data = items['results']
            else:
                # Treat the dict as a single item
                items_data = [items]
        else:
            st.error(f"Unexpected items type: {type(items)}")
            return pd.DataFrame()

        if items_data is None or not items_data:
            st.warning("No keyword data found in API response")
            return pd.DataFrame()

        # Ensure items_data is iterable
        if not hasattr(items_data, '__iter__'):
            st.error(f"Items data is not iterable: {type(items_data)}")
            return pd.DataFrame()

        for item in items_data:
            if isinstance(item, dict):
                info = item.get("keyword_info", item)  # Some APIs return data directly
                cpc = info.get("cpc", 0)
                
                # Handle different CPC formats
                if isinstance(cpc, dict):
                    cpc_value = cpc.get("cpc", 0)
                else:
                    cpc_value = cpc
                    
                rows.append({
                    "keyword": item.get("keyword", ""),
                    "search_volume": info.get("search_volume", 0),
                    "cpc_usd": cpc_value,
                    "competition": info.get("competition", 0),
                })
    
    except Exception as e:
        st.error(f"Error processing keyword suggestions response: {e}")
        if st.session_state.get('show_raw_data', False):
            st.write("Raw items:", items)
        return pd.DataFrame()
    
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner="Scanning site for keywords...")
def get_keywords_from_site(url: str, lang_code: str, loc_name: str, limit: int) -> pd.DataFrame:
    payload_item = {
        "target": url.strip(),
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
        "limit": limit,
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/keywords_for_site/live", post_data)
    
    if st.session_state.get('show_raw_data', False):
        st.json(response)
    
    items = extract_items_from_response(response)
    
    if not items:
        st.warning("No data returned from keywords for site API")
        return pd.DataFrame()

    rows = []
    try:
        # Debug: Check what we actually got
        if st.session_state.get('show_raw_data', False):
            st.write("Items structure:", type(items))
            st.write("Items content:", items)
        
        # Handle different possible response structures
        items_data = None
        
        if isinstance(items, list):
            if len(items) > 0:
                first_item = items[0]
                if isinstance(first_item, dict):
                    if 'items' in first_item:
                        items_data = first_item['items']
                    elif 'keyword' in first_item:  # Direct keyword data
                        items_data = items
                    else:
                        # Look for other possible data containers
                        for key in ['data', 'results', 'keywords']:
                            if key in first_item:
                                items_data = first_item[key]
                                break
                else:
                    items_data = items
            else:
                items_data = []
        elif isinstance(items, dict):
            # If items is a dict, look for keyword data
            if 'items' in items:
                items_data = items['items']
            elif 'data' in items:
                items_data = items['data']
            elif 'results' in items:
                items_data = items['results']
            else:
                # Treat the dict as a single item
                items_data = [items]
        else:
            st.error(f"Unexpected items type: {type(items)}")
            return pd.DataFrame()

        if items_data is None or not items_data:
            st.warning("No keyword data found in API response")
            return pd.DataFrame()

        # Ensure items_data is iterable
        if not hasattr(items_data, '__iter__'):
            st.error(f"Items data is not iterable: {type(items_data)}")
            return pd.DataFrame()

        for item in items_data:
            if isinstance(item, dict):
                # Try different possible field names for keyword info
                info = item.get("keyword_info", item.get("keyword_data", item))
                
                # Handle different CPC formats
                cpc = info.get("cpc", 0)
                if isinstance(cpc, dict):
                    cpc_value = cpc.get("cpc", 0)
                else:
                    cpc_value = cpc
                    
                rows.append({
                    "keyword": item.get("keyword", ""),
                    "search_volume": info.get("search_volume", 0),
                    "cpc_usd": cpc_value,
                    "competition": info.get("competition", 0),
                })
    
    except Exception as e:
        st.error(f"Error processing keywords from site response: {e}")
        if st.session_state.get('show_raw_data', False):
            st.write("Raw items:", items)
        return pd.DataFrame()
    
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner="Fetching metrics for your keywords...")
def get_keyword_metrics(keywords: list, lang_code: str, loc_name: str) -> pd.DataFrame:
    payload_item = {
        "keywords": keywords,
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/keywords_data/google_ads/search_volume/live", post_data)
    
    if st.session_state.get('show_raw_data', False):
        st.json(response)
    
    items = extract_items_from_response(response)

    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append({
                "keyword": item.get("keyword", ""),
                "search_volume": item.get("search_volume", 0),
                "cpc_usd": item.get("cpc", 0),
                "competition": item.get("competition", 0),
            })
    
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner="Analysing search intent...")
def get_search_intent(keywords: list, lang_code: str) -> pd.DataFrame:
    payload_item = {
        "keywords": keywords,
        "language_code": lang_code.strip()
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/search_intent/live", post_data)
    
    if st.session_state.get('show_raw_data', False):
        st.json(response)
    
    items = extract_items_from_response(response)

    if not items:
        return pd.DataFrame()

    rows = []
    # Handle different possible response structures
    items_data = items
    if isinstance(items, list) and len(items) > 0:
        if 'items' in items[0]:
            items_data = items[0]['items']
        else:
            items_data = items

    for item in items_data:
        if isinstance(item, dict):
            intent_info = item.get("keyword_intent", {})
            rows.append({
                "keyword_clean": (item.get("keyword") or "").lower().strip(),
                "intent": intent_info.get("label", "unknown"),
                "intent_probability": intent_info.get("probability", 0)
            })
    
    return pd.DataFrame(rows)

# Main analysis button
if st.button("Analyse Keywords", type="primary"):
    df_metrics = pd.DataFrame()

    if analysis_mode == "Generate from Seed Keyword" and seed_keyword:
        df_metrics = get_keyword_suggestions(seed_keyword, language_code, location_name, limit)
    elif analysis_mode == "Analyse My Keyword List" and uploaded_keywords:
        keyword_chunks = [uploaded_keywords[i:i + 1000] for i in range(0, len(uploaded_keywords), 1000)]
        results_list = []
        for chunk in keyword_chunks:
            chunk_result = get_keyword_metrics(chunk, language_code, location_name)
            if not chunk_result.empty:
                results_list.append(chunk_result)
        
        if results_list:
            df_metrics = pd.concat(results_list, ignore_index=True)
    elif analysis_mode == "Scan Website for Keywords" and target_url:
        df_metrics = get_keywords_from_site(target_url, language_code, location_name, limit)

    if df_metrics.empty:
        st.warning("Could not retrieve keyword metrics. Please check your inputs or try different keywords.")
        st.session_state.results = None
    else:
        # Clean and prepare data
        df_metrics = df_metrics.dropna(subset=['keyword'])
        df_metrics = df_metrics[df_metrics['keyword'].str.strip() != '']
        
        # Ensure numeric columns are properly converted
        df_metrics['search_volume'] = pd.to_numeric(df_metrics['search_volume'], errors='coerce').fillna(0)
        df_metrics['cpc_usd'] = pd.to_numeric(df_metrics['cpc_usd'], errors='coerce').fillna(0)
        df_metrics['competition'] = pd.to_numeric(df_metrics['competition'], errors='coerce').fillna(0)
        
        df_metrics['keyword_clean'] = df_metrics['keyword'].str.lower().str.strip()

        # Apply pre-filter negative keywords
        pre_negatives = parse_negative_keywords(pre_negatives_input)
        if pre_negatives:
            df_metrics, pre_filtered_count = apply_negative_keywords(df_metrics, pre_negatives, match_type)
            if pre_filtered_count > 0:
                st.success(f"Pre-filtered {pre_filtered_count} keywords based on your negative keyword list")

        intent_keywords = df_metrics['keyword_clean'].tolist()
        df_intent = get_search_intent(intent_keywords, language_code)

        if not df_intent.empty:
            df_merged = pd.merge(df_metrics, df_intent, on="keyword_clean", how="left")
            df_merged = df_merged.drop(columns=['keyword_clean'])
        else:
            df_merged = df_metrics.drop(columns=['keyword_clean'])
            df_merged['intent'] = 'unknown'
            df_merged['intent_probability'] = 0

        # Fill missing intents
        df_merged['intent'] = df_merged['intent'].fillna('unknown')
        
        st.session_state.results = {"df_merged": df_merged}

# Display results
if st.session_state.results:
    df_merged = st.session_state.results["df_merged"].copy()

    # Convert CPC to GBP
    df_merged["cpc_gbp"] = (df_merged["cpc_usd"] * usd_to_gbp_rate).round(2)

    # Apply filters if enabled
    if enable_filters:
        original_count = len(df_merged)

        # Search volume filters
        df_merged = df_merged[df_merged['search_volume'] >= min_search_volume]
        if max_search_volume > 0:
            df_merged = df_merged[df_merged['search_volume'] <= max_search_volume]

        # CPC filters
        df_merged = df_merged[df_merged['cpc_gbp'] >= min_cpc]
        if max_cpc > 0:
            df_merged = df_merged[df_merged['cpc_gbp'] <= max_cpc]

        # Competition filters
        df_merged = df_merged[df_merged['competition'] >= min_competition]
        df_merged = df_merged[df_merged['competition'] <= max_competition]

        # Intent filter
        if intent_filter:
            df_merged = df_merged[df_merged['intent'].isin(intent_filter)]

        filtered_count = len(df_merged)
        if filtered_count < original_count:
            st.info(f"Filters applied: {original_count - filtered_count} keywords removed, {filtered_count} remaining")

    # Calculate keyword difficulty and opportunity scores
    df_merged['difficulty_score'] = df_merged.apply(
        lambda row: calculate_keyword_difficulty(row['competition'], row['cpc_gbp']), axis=1
    ).round(1)

    df_merged['opportunity_score'] = df_merged.apply(
        lambda row: calculate_opportunity_score(
            row['search_volume'], row['cpc_gbp'], row['competition'], row.get('intent', 'unknown')
        ), axis=1
    ).round(1)

    # Add keyword grouping
    df_merged = group_keywords_by_similarity(df_merged)

    st.subheader("Keyword Analysis Results")

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Keywords", f"{len(df_merged):,}")
    with col2:
        avg_difficulty = df_merged['difficulty_score'].mean()
        st.metric("Avg Difficulty", f"{avg_difficulty:.1f}/100")
    with col3:
        avg_opportunity = df_merged['opportunity_score'].mean()
        st.metric("Avg Opportunity", f"{avg_opportunity:.1f}/100")
    with col4:
        total_vol = df_merged['search_volume'].sum()
        st.metric("Total Volume", f"{total_vol:,}")

    display_cols = ['keyword', 'search_volume', 'cpc_gbp', 'competition', 'intent', 'difficulty_score', 'opportunity_score', 'keyword_group']
    st.dataframe(df_merged[display_cols].sort_values('opportunity_score', ascending=False), use_container_width=True, height=400)

    # Post-Filter Negative Keywords
    st.divider()
    with st.expander("🚫 Add More Negative Keywords (Post-Filter)", expanded=False):
        st.caption("Refine results by excluding additional keywords. This filters the displayed data without re-running the API.")

        post_negatives_input = st.text_area(
            "Enter additional negative keywords to filter out",
            value="",
            height=80,
            placeholder="e.g., training, course, tutorial",
            key="post_negatives",
            help="Separate by commas or new lines"
        )

        post_match_col1, post_match_col2 = st.columns([1, 2])
        with post_match_col1:
            post_match_type = st.radio(
                "Match Type",
                ("Contains", "Exact Match"),
                key="post_match_type",
                help="Contains: Filter if keyword contains the term. Exact: Filter only if exact match."
            )

        with post_match_col2:
            if st.button("Apply Post-Filter", type="secondary"):
                post_negatives = parse_negative_keywords(post_negatives_input)
                if post_negatives:
                    df_temp, post_filtered_count = apply_negative_keywords(df_merged, post_negatives, post_match_type)
                    if post_filtered_count > 0:
                        # Update session state with filtered data
                        st.session_state.results["df_merged"] = df_temp
                        st.success(f"✅ Filtered out {post_filtered_count} keywords. {len(df_temp)} keywords remaining.")
                        st.info("🔄 Results updated! The analysis below now reflects your filters. Refresh the page or re-run to reset.")
                        st.rerun()  # Rerun to update all downstream calculations
                    else:
                        st.info("No keywords matched your negative filters.")
                else:
                    st.warning("Please enter at least one negative keyword.")

    # Quick Wins Section
    st.divider()
    st.subheader("Quick Wins - Low-Hanging Fruit")

    # Define quick wins: High opportunity (>60), Low difficulty (<40)
    quick_wins = df_merged[
        (df_merged['opportunity_score'] >= 60) &
        (df_merged['difficulty_score'] <= 40)
    ].sort_values('opportunity_score', ascending=False)

    if not quick_wins.empty:
        qw_col1, qw_col2 = st.columns([2, 1])

        with qw_col1:
            st.success(f"Found {len(quick_wins)} high-opportunity, low-difficulty keywords!")
            st.dataframe(
                quick_wins[['keyword', 'search_volume', 'cpc_gbp', 'intent', 'opportunity_score', 'difficulty_score']].head(20),
                use_container_width=True
            )

        with qw_col2:
            qw_metrics = quick_wins.agg({
                'search_volume': 'sum',
                'cpc_gbp': 'mean'
            })

            st.metric("Total Volume", f"{int(qw_metrics['search_volume']):,}")
            st.metric("Avg CPC", f"£{qw_metrics['cpc_gbp']:.2f}")

            # Estimate potential impact
            estimated_clicks = qw_metrics['search_volume'] * 0.03  # Conservative 3% CTR
            estimated_cost = estimated_clicks * qw_metrics['cpc_gbp']
            st.metric("Est. Monthly Clicks", f"{int(estimated_clicks):,}")
            st.metric("Est. Monthly Cost", f"£{estimated_cost:,.2f}")

        quick_wins_csv = quick_wins.to_csv(index=False).encode("utf-8")
        st.download_button("Download Quick Wins (CSV)", quick_wins_csv, "quick_wins_keywords.csv", "text/csv", key="qw_download")
    else:
        st.info("No quick wins identified with current filters. Try adjusting your criteria or filters.")

    # Competitive Insights
    st.divider()
    st.subheader("Competitive Insights")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        # High volume, high competition keywords (competitive battlefield)
        competitive_keywords = df_merged[
            (df_merged['search_volume'] >= df_merged['search_volume'].quantile(0.75)) &
            (df_merged['competition'] >= 0.7)
        ].sort_values('search_volume', ascending=False).head(10)

        st.write("**Highly Competitive Keywords** (High volume, High competition)")
        st.caption("These keywords are competitive but have significant traffic potential")
        if not competitive_keywords.empty:
            st.dataframe(
                competitive_keywords[['keyword', 'search_volume', 'cpc_gbp', 'competition', 'intent']],
                use_container_width=True
            )
        else:
            st.info("No highly competitive keywords found")

    with comp_col2:
        # Low competition gems
        low_comp_gems = df_merged[
            (df_merged['search_volume'] >= 50) &
            (df_merged['competition'] <= 0.3) &
            (df_merged['cpc_gbp'] >= 0.5)
        ].sort_values('search_volume', ascending=False).head(10)

        st.write("**Low Competition Gems** (Decent volume, Low competition)")
        st.caption("Easier to rank for with meaningful traffic")
        if not low_comp_gems.empty:
            st.dataframe(
                low_comp_gems[['keyword', 'search_volume', 'cpc_gbp', 'competition', 'intent']],
                use_container_width=True
            )
        else:
            st.info("No low competition gems found")

    # Debug unmatched keywords
    unmatched_keywords = df_merged[df_merged['intent'].isna() | (df_merged['intent'] == 'unknown')]['keyword'].tolist()
    if unmatched_keywords:
        with st.expander(f"Debug: {len(unmatched_keywords)} keywords could not be assigned an intent"):
            st.write(unmatched_keywords)

    # Create summary
    summary_df = df_merged[df_merged['intent'] != 'unknown'].dropna(subset=['intent'])

    if not summary_df.empty:
        summary = summary_df.groupby("intent").agg(
            keywords=("keyword", "count"),
            total_volume=("search_volume", "sum"),
            avg_cpc_gbp=("cpc_gbp", safe_average),
        ).reset_index().rename(columns={"intent": "Intent"})

        # Ensure all numeric columns are properly typed
        summary["total_volume"] = pd.to_numeric(summary["total_volume"], errors='coerce').fillna(0)
        summary["avg_cpc_gbp"] = pd.to_numeric(summary["avg_cpc_gbp"], errors='coerce').fillna(0)

        # Add CTR and CVR
        summary["CTR"] = summary["Intent"].map(ctrs).fillna(0.03)
        summary["CVR"] = summary["Intent"].map(cvrs).fillna(0.015)
        
        # Calculate metrics
        summary["Max Clicks"] = (summary["total_volume"] * summary["CTR"]).round(0)
        summary["Max Spend £"] = (summary["Max Clicks"] * summary["avg_cpc_gbp"]).round(2)
        
        required_budget = summary["Max Spend £"].sum()
        if not use_custom_budget:
            st.sidebar.metric("Required Budget", f"£{required_budget:,.2f}")

        if use_custom_budget:
            summary["Budget £"] = summary["Intent"].map(budget_allocations).fillna(25) * total_budget / 100
            summary["Clicks"] = np.where(
                summary["Budget £"] < summary["Max Spend £"],
                (summary["Budget £"] / summary["avg_cpc_gbp"]).round(0),
                summary["Max Clicks"]
            )
            summary["Spend £"] = (summary["Clicks"] * summary["avg_cpc_gbp"]).round(2)
        else:
            summary["Clicks"] = summary["Max Clicks"]
            summary["Spend £"] = summary["Max Spend £"]

        summary["Conversions"] = (summary["Clicks"] * summary["CVR"]).round(0)
        summary["CPA £"] = np.where(
            summary["Conversions"] > 0,
            (summary["Spend £"] / summary["Conversions"]).round(2),
            0
        )

        # Add ROI metrics if enabled
        if enable_roi_calc:
            summary = calculate_roi_metrics(summary, avg_order_value, profit_margin)

        st.subheader("Grouped by Search Intent")

        if enable_roi_calc:
            display_summary = summary[['Intent', 'keywords', 'total_volume', 'Clicks', 'Spend £', 'Conversions', 'CPA £', 'total_revenue', 'total_profit', 'ROAS', 'ROI']]
            display_summary = display_summary.rename(columns={
                'total_revenue': 'Revenue £',
                'total_profit': 'Profit £',
                'ROAS': 'ROAS',
                'ROI': 'ROI %'
            })
        else:
            display_summary = summary[['Intent', 'keywords', 'total_volume', 'Clicks', 'Spend £', 'Conversions', 'CPA £']]

        st.dataframe(display_summary.fillna("—"), use_container_width=True)

        # Show budget optimization if enabled
        if show_budget_optimization and use_custom_budget:
            st.subheader("Budget Optimization Recommendations")
            optimized = optimize_budget_allocation(summary, total_budget)

            st.write("**Current vs Optimized Allocation:**")
            comparison_df = optimized[['Intent', 'Budget £', 'optimized_budget', 'CPA £']].copy()
            comparison_df = comparison_df.rename(columns={
                'Budget £': 'Current Budget £',
                'optimized_budget': 'Recommended Budget £',
                'CPA £': 'Expected CPA £'
            })
            st.dataframe(comparison_df, use_container_width=True)

            st.info("Optimized budget allocation is based on inverse CPA - intents with lower CPA receive more budget for better efficiency.")

        # Blended overview
        total_keywords = summary["keywords"].sum()
        total_volume = summary["total_volume"].sum()
        total_clicks = summary["Clicks"].sum()
        total_spend = summary["Spend £"].sum()
        total_conversions = summary["Conversions"].sum()

        blended_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        blended_ctr = total_clicks / total_volume if total_volume > 0 else 0
        blended_cvr = total_conversions / total_clicks if total_clicks > 0 else 0
        blended_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        blended_overview = pd.DataFrame({
            "Total Keywords": [int(total_keywords)],
            "Total Volume": [int(total_volume)],
            "Weighted Avg CPC £": [round(blended_cpc, 2)],
            "Weighted CTR": [round(blended_ctr, 3)],
            "Total Clicks": [int(total_clicks)],
            "Weighted CVR": [round(blended_cvr, 3)],
            "Total Conversions": [int(total_conversions)],
            "Total Spend £": [round(total_spend, 2)],
            "Blended CPA £": [round(blended_cpa, 2)]
        })
        st.subheader("Blended Overview (Weighted)")
        st.dataframe(blended_overview, use_container_width=True)

        # Add ROI to blended overview if enabled
        if enable_roi_calc:
            total_revenue = summary['total_revenue'].sum() if 'total_revenue' in summary.columns else 0
            total_profit = summary['total_profit'].sum() if 'total_profit' in summary.columns else 0
            blended_roas = total_revenue / total_spend if total_spend > 0 else 0
            blended_roi = ((total_profit - total_spend) / total_spend * 100) if total_spend > 0 else 0

            roi_cols = st.columns(4)
            with roi_cols[0]:
                st.metric("Total Revenue", f"£{total_revenue:,.2f}")
            with roi_cols[1]:
                st.metric("Total Profit", f"£{total_profit:,.2f}")
            with roi_cols[2]:
                st.metric("ROAS", f"{blended_roas:.2f}")
            with roi_cols[3]:
                st.metric("ROI", f"{blended_roi:.1f}%")

        # Negative keywords suggestions
        st.divider()
        with st.expander("View Negative Keyword Suggestions", expanded=False):
            df_negatives = suggest_negative_keywords(df_merged)
            if not df_negatives.empty:
                st.write(f"Found {len(df_negatives)} keywords that may be good negative keyword candidates:")
                st.dataframe(df_negatives, use_container_width=True)

                neg_csv = df_negatives.to_csv(index=False).encode("utf-8")
                st.download_button("Download Negative Keywords (CSV)", neg_csv, "negative_keywords.csv", "text/csv", key="neg_download")
            else:
                st.success("No obvious negative keyword candidates found in your list!")

        # Keyword Groups Analysis
        st.divider()
        with st.expander("View Keyword Groups (for Ad Group Creation)", expanded=False):
            group_analysis = df_merged.groupby('keyword_group').agg({
                'keyword': 'count',
                'search_volume': 'sum',
                'cpc_gbp': 'mean',
                'opportunity_score': 'mean'
            }).reset_index()
            group_analysis.columns = ['Keyword Group', 'Keywords', 'Total Volume', 'Avg CPC £', 'Avg Opportunity']
            group_analysis = group_analysis.sort_values('Total Volume', ascending=False)

            st.write("Keywords grouped by common themes:")
            st.dataframe(group_analysis, use_container_width=True)

            # Show keywords in each group
            selected_group = st.selectbox("View keywords in group:", group_analysis['Keyword Group'].tolist())
            if selected_group:
                group_keywords = df_merged[df_merged['keyword_group'] == selected_group][['keyword', 'search_volume', 'cpc_gbp', 'intent', 'opportunity_score']]
                st.dataframe(group_keywords.sort_values('opportunity_score', ascending=False), use_container_width=True)

        # Visualisation
        st.divider()
        st.subheader("Data Visualizations")

        viz_tabs = st.tabs(["Intent Performance", "Opportunity Analysis", "Keyword Distribution"])

        with viz_tabs[0]:
            metric_mapping_clean = {
                "Total Volume": "total_volume",
                "Clicks": "Clicks",
                "Spend (£)": "Spend_GBP",
                "Conversions": "Conversions",
                "CPA (£)": "CPA_GBP"
            }

            if enable_roi_calc:
                metric_mapping_clean.update({
                    "Revenue (£)": "total_revenue",
                    "ROAS": "ROAS",
                    "ROI (%)": "ROI"
                })

            chart_metric_display = st.selectbox(
                "Choose a metric to visualise",
                list(metric_mapping_clean.keys())
            )

            chart_df = summary.copy()
            chart_df.columns = [col.replace('£', 'GBP').replace(' ', '_') for col in chart_df.columns]
            chart_metric_col = metric_mapping_clean[chart_metric_display]

            if chart_metric_col in chart_df.columns:
                chart = alt.Chart(chart_df).mark_bar(
                    color="#48d597"
                ).encode(
                    x=alt.X('Intent:N', sort='-y', title='Search Intent'),
                    y=alt.Y(f'{chart_metric_col}:Q', title=chart_metric_display),
                    tooltip=['Intent', alt.Tooltip(f'{chart_metric_col}:Q', title=chart_metric_display, format=',.0f')]
                ).properties(
                    title=f'{chart_metric_display} by Search Intent'
                ).configure_axis(
                    labelAngle=0
                ).configure_title(
                    fontSize=16
                )
                st.altair_chart(chart, use_container_width=True, theme=None)

        with viz_tabs[1]:
            # Opportunity vs Difficulty Scatter Plot
            st.write("**Opportunity vs Difficulty Analysis**")
            st.caption("Top-right quadrant = High opportunity, High difficulty | Top-left = High opportunity, Low difficulty (Sweet spot!)")

            scatter_data = df_merged[['keyword', 'opportunity_score', 'difficulty_score', 'search_volume', 'cpc_gbp', 'intent']].copy()

            scatter = alt.Chart(scatter_data).mark_circle(size=60).encode(
                x=alt.X('difficulty_score:Q', title='Difficulty Score', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('opportunity_score:Q', title='Opportunity Score', scale=alt.Scale(domain=[0, 100])),
                color=alt.Color('intent:N', title='Intent'),
                size=alt.Size('search_volume:Q', title='Search Volume', scale=alt.Scale(range=[50, 500])),
                tooltip=['keyword', 'opportunity_score', 'difficulty_score', 'search_volume', 'cpc_gbp', 'intent']
            ).properties(
                width=700,
                height=500,
                title='Keyword Opportunity vs Difficulty'
            ).interactive()

            # Add quadrant lines
            h_line = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(strokeDash=[5, 5], color='gray').encode(y='y:Q')
            v_line = alt.Chart(pd.DataFrame({'x': [50]})).mark_rule(strokeDash=[5, 5], color='gray').encode(x='x:Q')

            st.altair_chart(scatter + h_line + v_line, use_container_width=True, theme=None)

            # Top opportunities
            st.write("**Top 10 High-Opportunity Keywords**")
            top_opps = df_merged.nlargest(10, 'opportunity_score')[['keyword', 'search_volume', 'cpc_gbp', 'difficulty_score', 'opportunity_score', 'intent']]
            st.dataframe(top_opps, use_container_width=True)

        with viz_tabs[2]:
            # Distribution charts
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Search Volume Distribution**")
                vol_chart = alt.Chart(df_merged).mark_bar(color="#48d597").encode(
                    alt.X('search_volume:Q', bin=alt.Bin(maxbins=20), title='Search Volume'),
                    alt.Y('count():Q', title='Number of Keywords')
                ).properties(height=300)
                st.altair_chart(vol_chart, use_container_width=True, theme=None)

            with col2:
                st.write("**CPC Distribution**")
                cpc_chart = alt.Chart(df_merged).mark_bar(color="#48d597").encode(
                    alt.X('cpc_gbp:Q', bin=alt.Bin(maxbins=20), title='CPC (£)'),
                    alt.Y('count():Q', title='Number of Keywords')
                ).properties(height=300)
                st.altair_chart(cpc_chart, use_container_width=True, theme=None)

            st.write("**Intent Distribution**")
            intent_dist = df_merged['intent'].value_counts().reset_index()
            intent_dist.columns = ['Intent', 'Count']

            intent_pie = alt.Chart(intent_dist).mark_arc().encode(
                theta=alt.Theta('Count:Q'),
                color=alt.Color('Intent:N'),
                tooltip=['Intent', 'Count']
            ).properties(height=300)
            st.altair_chart(intent_pie, use_container_width=True, theme=None)

        # Download buttons
        st.divider()
        st.subheader("Export Data")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            csv_data = df_merged.to_csv(index=False).encode("utf-8")
            st.download_button("Download Detailed Data (CSV)", csv_data, "keyword_intent_details.csv", "text/csv", key="d1")

        with col2:
            summary_csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button("Download Intent Summary (CSV)", summary_csv, "intent_summary.csv", "text/csv", key="d2")

        with col3:
            overview_csv = blended_overview.to_csv(index=False).encode("utf-8")
            st.download_button("Download Blended Overview (CSV)", overview_csv, "blended_overview.csv", "text/csv", key="d3")

        with col4:
            # Excel export with multiple sheets
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_merged.to_excel(writer, sheet_name='All Keywords', index=False)
                summary.to_excel(writer, sheet_name='Intent Summary', index=False)
                blended_overview.to_excel(writer, sheet_name='Blended Overview', index=False)

                if not df_negatives.empty:
                    df_negatives.to_excel(writer, sheet_name='Negative Keywords', index=False)

                group_analysis.to_excel(writer, sheet_name='Keyword Groups', index=False)

            excel_buffer.seek(0)
            st.download_button(
                "Download Full Report (Excel)",
                excel_buffer,
                "keyword_analysis_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )

    else:
        st.warning("Could not generate intent summary as no intent data was returned.")

    # Cost estimation
    if not df_merged.empty:
        num_keywords = len(df_merged)
        cost_sug = 0.01 + num_keywords * 0.0001
        cost_int = 0.001 + num_keywords * 0.0001
        approx_cost = cost_sug + cost_int
        st.caption(f"Approximate API cost for this run: ${approx_cost:.4f} for {num_keywords} keywords (estimate only).")
