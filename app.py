import streamlit as st
from bs4 import BeautifulSoup
import json
import requests 
import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import WebDriverException, TimeoutException
import time
from datetime import datetime
import traceback
import io
from PIL import Image
import cv2
import asyncio
import aiohttp
import urllib.parse
import os
import random # Added import
from google import genai 
from urllib.parse import urlparse, parse_qs, unquote # Import necessary functions
from langdetect import detect
import numpy as np
from tokencost import count_string_tokens
import imagehash
from google.genai import types
import gc
from GoogleAds.main import GoogleAds, show_regions_list
from urllib.parse import urlparse, parse_qs

import re
st.set_page_config(layout="wide",page_title= "Google Scrape", page_icon="🥽")
a = GoogleAds()
def get_secret(key: str, default=None):
    """Fetch a secret from Streamlit or fall back to environment variables."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


# --- Gemini Import and Configuration ---
api_keys_str = get_secret("GEMINI_API_KEY", "")

if api_keys_str:
    try:
        # Support either a JSON array of keys or a single key string
        parsed = json.loads(str(api_keys_str))
        GEMINI_API_KEYS = parsed if isinstance(parsed, list) else [parsed]
        
    except json.JSONDecodeError:
        GEMINI_API_KEYS = api_keys_str
else:
    st.warning(
        "GEMINI_API_KEY not found in Streamlit secrets or environment. Gemini functionality will be disabled.",
        icon="⚠️",
    )
    GEMINI_API_KEYS = None
# --- Gemini Function ---
def gemini_text_lib(prompt, model='gemini-2.5-pro-exp-03-25',max_retries=5): # Using a stable model  
    tries = 0
    while tries < max_retries:
        
        st.text(f"Gemini working.. {model} trial {tries+1}")
        """ Calls Gemini API, handling potential list of keys """
        if not GEMINI_API_KEYS:
            st.error("Gemini API keys not available.")
            return None
    
        # If multiple keys, choose one randomly; otherwise use the configured one (if single) or the first.
        selected_key = random.choice(GEMINI_API_KEYS)
    
        client = genai.Client(api_key=selected_key)
    
        config =types.GenerationConfig(temperature=0.7)
        
        
        try:
            response = client.models.generate_content(
                model=model, contents=  prompt
                
            )
            # st.text(str(response))
    
            return response.text
        except Exception as e:
            st.text('gemini_text_lib error ' + str(e)) 
            time.sleep(15)
            tries += 1
    
    return None

def get_top_3_media_hashes(media_list):
    """
    Processes media to find the top 3 most common perceptual hashes.
    For videos, it only analyzes the very first frame.

    Args:
        media_list (list): A list of strings, where each string is an image URL,
                           a video URL, or a local path to a video file.

    Returns:
        list: A list of tuples, sorted by frequency, containing the top 3
              hashes and their associated data.
    """
    hashes_map = {}

    def add_hash(img, source_identifier):
        """Helper function to compute hash and update the map."""
        try:
            h = imagehash.phash(img)
            hash_str = str(h)
            entry = hashes_map.get(hash_str, {'count': 0, 'data': []})
            entry['count'] += 1
            entry['data'].append(source_identifier)
            hashes_map[hash_str] = entry
        except Exception as e:
            print(f"Could not hash media from {source_identifier}: {e}")

    def process_video_stream(video_source):
        """
        Downloads the video (if it's a URL), and hashes the first frame.
        """
        try:
            local_path = None

            if isinstance(video_source, str) and video_source.startswith("http"):
                # Download video to temp file with headers
                headers = {
                    'User-Agent': (
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/115.0.0.0 Safari/537.36'
                    ),
                    'Referer': 'https://www.facebook.com/',
                }

                res = requests.get(video_source, headers=headers, stream=True, timeout=30)
                res.raise_for_status()

                # Save to temporary file
                local_path = f"_tmp_q.mp4"
                with open(local_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                local_path = video_source

            # Now open locally
            cap = cv2.VideoCapture(local_path)
            if not cap.isOpened():
                print(f"Error opening video: {video_source}")
                return

            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                add_hash(img, video_source)
            else:
                print(f"Failed to read frame from {video_source}")

            cap.release()

            # Clean up if we downloaded
            if video_source.startswith("http") and os.path.exists(local_path):
                os.remove(local_path)

        except Exception as e:
            print(f"Failed to process video {video_source}: {e}")


    for item in media_list:
        # --- Handle URLs (Image or Video) ---
        if item.startswith(('http://', 'https://')):
            try:
                headers = {
                    'User-Agent': (
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/115.0.0.0 Safari/537.36'
                    ),
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.facebook.com/',
                    'Connection': 'keep-alive'
}


                head_res = requests.head(item, timeout=11, allow_redirects=True,headers=headers)
                head_res.raise_for_status()
                content_type = head_res.headers.get('Content-Type', '').lower()

                if 'image' in content_type:
                    res = requests.get(item, timeout=20)
                    res.raise_for_status()
                    img = Image.open(io.BytesIO(res.content))
                    add_hash(img, item)
                
                elif 'video' in content_type:
                    process_video_stream(item)
                
                else:
                    print(f"Skipping unsupported URL content type '{content_type}' for: {item}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to access URL {item}: {e}")

        # --- Handle Local Files ---
        elif os.path.exists(item):
            process_video_stream(item)

        else:
            print(f"Skipping invalid item (not a URL or existing file): {item}")

    # Sort and return the top 3
    top3_most_common = sorted(
        hashes_map.items(),
        key=lambda k: k[1]['count'],
        reverse=True
    )[:3]
    return top3_most_common


# --- Memory Utilities ---
def _bytes_to_readable(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def optimize_dataframe_memory(df: pd.DataFrame, category_threshold: float = 0.5) -> pd.DataFrame:
    """Downcast numerics and convert low-cardinality object columns to category."""
    if df is None or df.empty:
        return df

    df_opt = df.copy(deep=False)

    # Downcast numeric columns
    for col in df_opt.select_dtypes(include=["int", "int64", "float"]).columns:
        try:
            if pd.api.types.is_float_dtype(df_opt[col]):
                df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
            else:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
        except Exception:
            pass

    # Convert object columns with low cardinality to category
    for col in df_opt.select_dtypes(include=["object"]).columns:
        try:
            n_unique = df_opt[col].nunique(dropna=False)
            n_total = len(df_opt[col]) if len(df_opt[col]) else 1
            if (n_unique / max(n_total, 1)) <= category_threshold:
                df_opt[col] = df_opt[col].astype("category")
        except Exception:
            pass

    return df_opt


def compact_final_results(df: pd.DataFrame, truncate_max_text_to: int = 240) -> pd.DataFrame:
    """Drop heavy columns and shrink strings for the final results table."""
    if df is None or df.empty:
        return df

    df_light = df.copy(deep=False)

    # Drop heavy rarely-used columns
    # for col in ["texts", "indices", "images"]:
    #     if col in df_light.columns:
    #         try:
    #             df_light.drop(columns=[col], inplace=True)
    #         except Exception:
    #             pass

    # Shorten verbose text to reduce memory and UI payload
    if "max_text" in df_light.columns:
        try:
            df_light["max_text"] = df_light["max_text"].astype(str).str.slice(0, truncate_max_text_to)
        except Exception:
            pass

    # Ensure boolean dtype for selection
    if "selected" in df_light.columns:
        try:
            df_light["selected"] = df_light["selected"].fillna(False).astype(bool)
        except Exception:
            pass

    # Apply general optimization
    df_light = optimize_dataframe_memory(df_light, category_threshold=0.7)

    return df_light




# --- Initialize Session State ---
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None # Initialize as None
if 'final_merged_df' not in st.session_state:
    st.session_state.final_merged_df = None


# --- Scraping Function (scrape_facebook_ads) ---
# [NO CHANGES NEEDED TO THE scrape_facebook_ads function itself from the previous version]
# ... (Keep the entire function as it was) ...
# # def scrape_facebook_ads(url, search_term, scroll_pause_time=5, max_scrolls=50):
#     """
#     Scrapes ads from a given Facebook Ads Library URL using Selenium (Cloud-ready).

#     Args:
#         url (str): The specific Facebook Ads Library URL to scrape (with q= term).
#         search_term (str): The search term used for this specific scrape run.
#         scroll_pause_time (int): Base pause time between scrolls.
#         max_scrolls (int): Maximum number of scroll attempts.

#     Returns:
#         pandas.DataFrame: A DataFrame containing the scraped ad data, or None if error.
#         list: Status messages.
#     """
#     status_messages = []
#     ads_data = []
#     driver = None

#     status_messages.append(f"Attempting to initialize WebDriver for term: '{search_term}' in cloud environment...")
#     try:
#         options = Options()
#         options.add_argument("--headless")  # Run headless REQUIRED for Streamlit Cloud
#         options.add_argument("--no-sandbox")  # REQUIRED
#         options.add_argument("--disable-dev-shm-usage")  # REQUIRED
#         options.add_argument("--disable-gpu") # Also often recommended
#         options.add_argument("--window-size=1920,1080") # Can be helpful
#         options.add_argument('--log-level=3') # Suppress logs

#         # In Streamlit Cloud, Selenium should automatically find chromedriver
#         # if it's installed via packages.txt and in the PATH.
#         # We try initializing without specifying executable_path.
#         try:
#              # Let Selenium handle the driver path if installed via packages.txt
#              # No explicit 'service' needed if chromedriver is in PATH
#              driver = webdriver.Chrome(options=options)
#              status_messages.append("WebDriver initialized successfully using system PATH.")
#         except WebDriverException as e:
#              status_messages.append(f"WebDriver auto-init failed: {e}. Trying with default Service()...")
#              # Fallback: Sometimes explicitly using Service() helps Selenium find it
#              try:
#                  service = Service() # Initialize without path
#                  driver = webdriver.Chrome(service=service, options=options)
#                  status_messages.append("WebDriver initialized successfully using default Service().")
#              except WebDriverException as e2:
#                  status_messages.append(f"WebDriver explicit Service() failed: {e2}")
#                  # Updated message based on packages.txt correction
#                  status_messages.append("Ensure 'chromium' and 'chromium-driver' are in packages.txt")
#                  st.error("Fatal: Could not initialize WebDriver in the cloud environment. Check packages.txt.")
#                  return None, status_messages # Critical failure

#         status_messages.append(f"Loading URL for '{search_term}': {url}")
#         driver.get(url)
#         driver.implicitly_wait(10) # Give page elements time to appear
#         time.sleep(5) # Initial wait after load

#         # --- Scrolling Logic ---
#         # (Keep scrolling logic as before, but add more robust waits/error handling)
#         screen_height = driver.execute_script("return window.screen.height;")
#         last_height = driver.execute_script("return document.body.scrollHeight")
#         scroll_count = 0
#         status_messages.append(f"Starting scroll process for '{search_term}'...")
#         scroll_status_placeholder = st.empty()

#         while True:
#             try:
#                 # Scroll down
#                 driver.execute_script(f"window.scrollTo(0, {last_height + screen_height});")
#                 time.sleep(0.5) # Short pause between scrolls
#                 driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight);") # Try to ensure bottom
#                 scroll_count += 1
#                 wait_time = scroll_pause_time + (scroll_count * 0.1) # Dynamic wait
#                 scroll_status_placeholder.info(f"Term '{search_term}': Scroll attempt {scroll_count}, waiting {wait_time:.1f}s...")
#                 time.sleep(wait_time)

#                 # Check new height
#                 new_height = driver.execute_script("return document.body.scrollHeight")
#                 if new_height == last_height:
#                     # Sometimes content loads just after height stabilizes, try one more time
#                     scroll_status_placeholder.info(f"Term '{search_term}': Height stable, checking one last time...")
#                     time.sleep(scroll_pause_time * 1.5) # Longer wait
#                     new_height = driver.execute_script("return document.body.scrollHeight")
#                     if new_height == last_height:
#                         status_messages.append(f"Reached end of page for '{search_term}'.")
#                         break

#                 last_height = new_height

#                 if scroll_count >= max_scrolls:
#                     status_messages.append(f"Reached max scroll attempts ({max_scrolls}) for '{search_term}'.")
#                     break

#             except TimeoutException:
#                  status_messages.append(f"Warning: Timeout during scroll execution for '{search_term}'. Page might be slow or stuck.")
#                  time.sleep(scroll_pause_time * 2) # Longer wait after timeout
#             except WebDriverException as scroll_err:
#                  status_messages.append(f"Warning: WebDriver error during scroll for '{search_term}': {scroll_err}. Trying to continue...")
#                  time.sleep(scroll_pause_time)


#         scroll_status_placeholder.empty()
#         status_messages.append(f"Scrolling finished for '{search_term}'.")

#         # Get page source
#         status_messages.append(f"Getting page source for '{search_term}'...")
#         html = driver.page_source
#         if not html or len(html) < 500: # Basic check for empty or minimal HTML
#              status_messages.append(f"Warning: Page source seems empty or too small for '{search_term}'. Check if the page loaded correctly.")
#              # Decide whether to continue or return early based on severity

#         status_messages.append(f"Parsing HTML for '{search_term}'...")
#         soup = BeautifulSoup(html, "lxml")

#         # --- Data Extraction Logic ---
#         # (Keep extraction logic as before - selectors remain the fragile part)
#         ad_block_selector = 'div.xh8yej3' # VERIFY THIS SELECTOR REGULARLY
#         ad_blocks = soup.select(ad_block_selector)
#         status_messages.append(f"Found {len(ad_blocks)} potential ad blocks for '{search_term}'.")

#         # ... (rest of the extraction loop is identical to previous version) ...
#         extraction_count = 0
#         for i, ad_block in enumerate(ad_blocks):
#             # (Same extraction logic for status, text, media_url)
#             # ...
#             status = "Not Found" # Placeholder
#             ad_text = "Not Found" # Placeholder
#             media_url = "Not Found" # Placeholder
#             count = 1
#             # --- [INSERT EXACT EXTRACTION CODE FROM PREVIOUS VERSION HERE] ---
#              # --- Extract Status ---
#             try:
#                 status_selectors = [
#                     'span.x1fp01tm', 'div[role="button"] > span', 'div > span[dir="auto"] > span[dir="auto"]'
#                 ]
#                 # Simplified status extraction (take first non-empty match)
#                 for selector in status_selectors:
#                     elem = ad_block.select_one(selector)
#                     if elem and elem.text.strip():
#                         status = elem.text.strip()
#                         break
#             except Exception: pass # Ignore errors in finding elements

#             # --- Extract Ad Text ---
#             try:
#                 text_selectors = [
#                     'div[data-ad-preview="message"]', 'div._7jyr',
#                     'div > div > span[dir="auto"]', 'div[style*="text-align"]'
#                 ]
#                 for selector in text_selectors:
#                     elem = ad_block.select_one(selector)
#                     if elem:
#                         all_texts = elem.find_all(string=True, recursive=True)
#                         full_text = ' '.join(filter(None, (t.strip() for t in all_texts)))
#                         cleaned_text = ' '.join(full_text.split())
#                         if cleaned_text and cleaned_text.lower() not in ["sponsored", "suggested for you", ""]:
#                             ad_text = cleaned_text
#                             break # Found good text
#                 if ad_text in ["Not Found", ""]: ad_text = "Not Found"
#             except Exception: pass

#             # --- Extract Page ID ---
#             try:
#                 text_selectors = [
#                         'a.xt0psk2.x1hl2dhg.xt0b8zv.x8t9es0.x1fvot60.xxio538.xjnfcd9.xq9mrsl.x1yc453h.x1h4wwuj.x1fcty0u']
#                 for selector in text_selectors:
#                     elem = ad_block.select_one(selector)
#                     if elem:
#                         page_name = elem.find_all(string=True, recursive=True)
#                         page_name =list(page_name)[0]
#                         page_id = elem.get("href", "Not Found")
#                         page_id = page_id.split("/")[3]

#                 if page_id in ["Not Found", ""]: page_id = "Not Found"
#                 if page_name in ["Not Found", ""]: page_id = "Not Found"

#                 # page_id = str(page_id)
#             except Exception:
#                 page_id ='fail'
#                 page_name = 'fail'

# # --- Extract count Text ---
#             try:
#                 count_selectors = [
#                    'div.x6s0dn4.x78zum5.xsag5q8 span[dir="auto"]', # Specific parent + specific span type
#                  'div.x6s0dn4.x78zum5.xsag5q8 span', # General span  
#                 ]
#                 for selector in count_selectors:
#                     elem = ad_block.select_one(selector)
#                     if elem:
#                         all_texts = elem.find_all(string=True, recursive=True)
#                         full_text = ' '.join(filter(None, (t.strip() for t in all_texts)))
#                         cleaned_text = ' '.join(full_text.split())
#                         if cleaned_text and cleaned_text.lower() not in ["sponsored", "suggested for you", ""]:
#                             count = int("".join(filter(str.isdigit, cleaned_text)))
#                             break # Found good text
#                 if count in ["Not Found", ""]: ad_text = "Not Found"
#             except Exception: pass

#             # --- Extract Image or Video Poster URL ---
#             try:
#                 media_url = "Not Found"
#                 img_selectors = [
#                     'img.x168nmei', 'img.xt7dq6l', 'img[referrerpolicy="origin-when-cross-origin"]',
#                     'div[role="img"] > img', 'img:not([width="16"]):not([height="16"])'
#                 ]
#                 # Look for images
#                 for selector in img_selectors:
#                      img_elem = ad_block.select_one(selector)
#                      if img_elem and img_elem.has_attr('src'):
#                          src = img_elem['src']
#                          if 'data:image' not in src and '/emoji.php/' not in src and 'static.xx.fbcdn.net/rsrc.php' not in src and "s60x60" not in src:
#                              media_url = src
#                              break
#                 # Look for video posters if no image found
#                 if media_url == "Not Found" or 1==1:
#                     video_selectors = [
#                         'video.x1lliihq', 'video.xvbhtw8', 'video[poster]'
#                     ]
#                     for selector in video_selectors:
#                          vid_elem = ad_block.select_one(selector)
#                         #  st.text(vid_elem)
#                          if vid_elem and vid_elem.has_attr('src'):
#                             media_url = vid_elem['src']
#                             break



                


#             except Exception: pass
#             # --- Extract Ad Link using CSS Selectors ---
#             ad_link = "Not Found"
#             found_link_tag = None
#             # Define potential CSS selectors for the link, ordered from most specific/reliable to more general
#             link_selectors = [
#                 'a[href^="https://l.facebook.com/l.php?u="]', # Starts with FB redirect + contains domain
#                 'div._7jyr + a[target="_blank"]', # Positional selector + target attribute
#                 'a[data-lynx-mode="hover"]',       # Just the data attribute
#                 'a[href^="https://l.facebook.com/l.php?u="]' # Just the FB redirect start
#             ]

#             try:
#                 for selector in link_selectors:
#                     # print(f"Trying selector: {selector}") # Optional debug print
#                     elem = ad_block.select_one(selector)
#                     # Check if an element was found and if it has an 'href' attribute
#                     if elem and elem.has_attr('href'):
#                         # print(f"Selector matched: {selector}") # Optional debug print
#                         found_link_tag = elem
#                         ad_link = found_link_tag['href'] # Extract the href value


#                         if ad_link and ad_link != "Not Found" and "l.facebook.com/l.php" in ad_link:
#                             try:
#                                 parsed_url = urlparse(ad_link)
#                                 query_params = parse_qs(parsed_url.query)
                                
#                                 # Check if the 'u' parameter exists
#                                 if 'u' in query_params:
#                                     # parse_qs returns a list for each param, get the first value
#                                     encoded_url = query_params['u'][0]
#                                     # Decode the URL
#                                     ad_link = unquote(encoded_url)
#                                 else:
#                                     ad_link = "Redirect link found, but 'u' parameter missing."
                                    
#                             except Exception as e:
#                                 print(f"Error parsing or decoding redirect URL: {e}")
#                                 actual_destination_url = "Error processing redirect link"
#                         elif ad_link and ad_link != "Not Found":
#                             # If it wasn't a facebook redirect link, the extracted link is the actual one
#                             ad_link = ad_link
#                         break # Stop searching once a link is found
#             except Exception as e:
#                 print(f"An error occurred while selecting the link: {e}")
#                 ad_link = "Error finding link"
#             # --- [END OF EXTRACTION CODE] ---


#             # Append data - include the search_term
#             # ** NOTE: Filtering based on Text/Media presence is now done AFTER collecting all rows **
#             ads_data.append({ 
#                  'Search_Term': search_term,
#                  'Status': status,
#                  'Text': ad_text,
#                  'Count': count,
#                  'Media_URL': media_url,
#                  'Landing_Page': ad_link,
#                  'Page ID' :page_id,
#                  'Page Name' : page_name
#              })
#             extraction_count += 1 # Count raw extracted rows

#         # --- End of Extraction Loop ---

#         final_message = f"Extracted {extraction_count} raw ad data entries for term '{search_term}' (before filtering)."
#         status_messages.append(final_message)

#         if ads_data:
#             df = pd.DataFrame(ads_data)
#             # Note: Filtering is now applied AFTER concatenation in the main app logic
#             return df, status_messages # Return the unfiltered data for this term
#         else:
#             status_messages.append(f"No raw data extracted into DataFrame for '{search_term}'.")
#             return pd.DataFrame(), status_messages # Return empty DF

#     except WebDriverException as e:
#         error_msg = f"WebDriver Error during operation for term '{search_term}': {e}\n{traceback.format_exc()}"
#         status_messages.append(error_msg)
#         st.error(f"WebDriver Error occurred for '{search_term}'. Check logs. The cloud environment might be unstable or the page blocked.")
#         return None, status_messages # Indicate failure
#     except Exception as e:
#         error_msg = f"Unexpected Error for term '{search_term}': {e}\n{traceback.format_exc()}"
#         status_messages.append(error_msg)
#         st.error(f"Unexpected Error occurred for '{search_term}'. Check logs.")
#         return None, status_messages # Indicate failure

#     finally:
#         if driver:
#             status_messages.append(f"Closing WebDriver for '{search_term}'...")
#             try:
#                 driver.quit()
#                 status_messages.append(f"WebDriver closed for '{search_term}'.")
#             except Exception as quit_err:
#                  status_messages.append(f"Error closing WebDriver for '{search_term}': {quit_err}")


def scrape_google_ads(term, max_creatives = 200):

    ads_data= []
    creatives = a.get_creative_Ids(term, max_creatives) # Get 200 creatives if available
    if creatives["Ad Count"]:
        advertisor_id = creatives["Advertisor Id"]
        for creative_id in creatives["Creative_Ids"]:
            try:
                details_json =a.get_detailed_ad(advertisor_id,creative_id)
                print(details_json)
                ad_title = details_json['Ad Title']
                landing_page = details_json['Ad Link']
                if details_json['Ad Title'] is '':
                    # st.text('souping')
                    req = requests.get(details_json['Ad Link'])
                    html = req.text
                    soup = BeautifulSoup(html, "html.parser")
                    elem = soup.find("a", attrs={"data-asoch-targets": re.compile(r"ad0.*title|title.*ad0", re.I)})
                    ad_title = elem.get_text(separator=" ", strip=True)
                    # st.text('title' + ad_title)
                    redirect_link = elem["href"]
                    url_parse = urlparse(redirect_link)
                    qs = parse_qs(url_parse.query)
                    landing_page = qs['adurl'][0]
            except:
                pass


            ads_data.append({ 
                 'Search_Term': term,
                #  'Status': status,
                 'Text': ad_title,
                 'Count': 1,
                 'Media_URL': details_json['Ad Link'],
                 'Landing_Page': landing_page,
                #  'Page ID' :page_id,
                #  'Page Name' : page_name
                "Last_Shown" : details_json['Last Shown']
             })

    if len(ads_data) > 0:
        return pd.DataFrame(ads_data)





# Fetch a single title with a semaphore (concurrency limiter)
async def fetch_title(session, url, semaphore, timeout=10):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                text = await resp.text()
                soup = BeautifulSoup(text, 'html.parser')
                title_tag = soup.find('title')
                return url, title_tag.get_text(strip=True) if title_tag else "[No title found]"
        except Exception as e:
            return url, f"[Error: {e}]"

# Limit concurrency using a semaphore
async def fetch_all_titles(urls, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_title(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)
        return results



def get_html_content(url):
    # Set up headless browser
    options = Options()
    options.add_argument("--headless")  # Run headless REQUIRED for Streamlit Cloud
    options.add_argument("--no-sandbox")  # REQUIRED
    options.add_argument("--disable-dev-shm-usage")  # REQUIRED
    options.add_argument("--disable-gpu") # Also often recommended
    options.add_argument("--window-size=1920,1080") # Can be helpful
    options.add_argument('--log-level=3') # Suppress logs
    # options.headless = True
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(2)  # Let JS load

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # # Keep only allowed tags
        # allowed_tags = ['a', 'p', 'h1', 'h2', 'h3', 'h4', 'li', 'ul', 'img']
        # for tag in soup.find_all(True):
        #     if tag.name not in allowed_tags:
        #         tag.decompose()

        # # Clean unwanted attributes
        # for tag in soup.find_all(allowed_tags):
        #     tag.attrs = {k: v for k, v in tag.attrs.items() if k in ['href', 'src', 'alt']}
        for tag in ["style", "link", "meta", "header", "footer", "input", "svg", "script"]:


            for element in soup.find_all(tag):
                element.decompose()
        
        # Optional: also remove iframes, noscript, or style if needed
        for tag in soup(['iframe', 'noscript']):
            tag.decompose()

        return str(soup)

    except Exception as e:
        st.text("error in get_html_content" + e)
        return None
    finally:
        driver.quit()

# --- Streamlit App UI ---
st.title("Google Ads Library Scraper")
st.markdown("""
Provide Base URL & Search Terms. Scrapes ads in the cloud, combines results, **filters for ads with Text & Media**, displays them, and optionally analyzes trends with Gemini.
""")

# --- Inputs ---
st.subheader("Configuration")
# mode = st.radio("Select Search Mode",["General Search","Page Search"] , index=0)

# if mode == 'General Search':
#     default_base_url = "https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country=ALL&is_targeted_country=false&media_type=all&search_type=keyword_exact_phrase&q="
#     base_url_template = st.text_input(
#         "Enter Base URL Template (ending with 'q=' or ready for term):",
#         default_base_url,
#         help="Example: https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=ALL&q="
#     )
# if mode == 'Page Search':
#     default_base_url = "https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country=ALL&is_targeted_country=false&media_type=all&search_type=page&view_all_page_id="
#     base_url_template = st.text_input(
#         "Enter Base URL Template (ending with 'view_all_page_id=' or ready for term):",
#         default_base_url,
#     )


search_terms_input = st.text_area(
    "Enter Search Terms\Page IDs (one per line):",
    height=350,
    help="Each line is a separate search query."
)
auto_gemini = st.checkbox("Auto Gemini Analyze?", value=False)
hash_imgs = st.checkbox("Analyze images and vid hash?", value=True)

st.info("ℹ️ WebDriver configured for Streamlit Cloud.", icon="☁️")

col1, col2 = st.columns(2)
with col1:
    max_creatives = st.slider("Max creatives to pull", min_value=1, max_value=500, value=200)



# --- Scrape Button and Logic ---
if st.button("🚀 Scrape All Terms in Cloud", type="primary"):
    # --- [Identical validation logic as before] ---
    if not search_terms_input:
        st.error("Please enter at least one search term.")
    else:
        search_terms = list(dict.fromkeys(term.strip() for term in search_terms_input.splitlines() if term.strip()))
        if not search_terms:
             st.error("No valid search terms found.")
        else:
            st.info(f"Preparing to scrape for {len(search_terms)} terms...")
            all_results_dfs = []
            count_total=0
            all_log_messages = []
            overall_start_time = time.time()
            overall_status_placeholder = st.empty()

            # --- [Identical scraping loop as before] ---
            for i, term in enumerate(search_terms):
                term_start_time = time.time()
                overall_status_placeholder.info(f"Processing term {i+1}/{len(search_terms)}: '{term}'... total scraped : {count_total}")
                encoded_term = urllib.parse.quote_plus(term)
            





                with st.spinner(f"Scraping '{term}'..." ):
                    scraped_df = scrape_google_ads(
                         term, max_creatives)
                    
                # st.text(scraped_df.to_csv)
                term_duration = time.time() - term_start_time
                if scraped_df is not None: # Check for None (fatal error)
                    if not scraped_df.empty:
                         # Append even if empty, filtering happens after concat
                        all_results_dfs.append(scraped_df)
                        count_total += len(scraped_df)
                    # Don't display success message per term here, do it after combining/filtering
                else:
                    st.error(f"Scraping failed for term '{term}' after {term_duration:.2f}s.")

            overall_status_placeholder.empty()
            overall_duration = time.time() - overall_start_time
            st.info(f"Finished scraping all {len(search_terms)} terms in {overall_duration:.2f} seconds. Now combining and filtering...")

            # --- Combine, Filter, and Store in Session State ---
            if all_results_dfs:
                # Combine all collected data (including potentially empty DFs from terms with no results)
                combined_raw_df = pd.concat(all_results_dfs, ignore_index=True)

                # Apply filtering HERE, after combining
                combined_filtered_df = combined_raw_df[
                    (combined_raw_df['Text'] != "Not Found") &
                    (combined_raw_df['Media_URL'] != "Not Found")
                ].copy() # Apply the filter condition from your code

                combined_filtered_df.reset_index(drop=True, inplace=True)

                if not combined_filtered_df.empty:
                    st.success(f"Combined and filtered data: {len(combined_filtered_df)} ads found with Text & Media.")
                    # Store the *filtered* DataFrame in session state
                    st.session_state.combined_df = combined_filtered_df
                else:
                    st.warning("No ads with both Text and Media found across all terms after filtering.")
                    st.session_state.combined_df = pd.DataFrame() # Store empty DF
            else:
                 st.warning("No data was scraped from any term.")
                 st.session_state.combined_df = None # Ensure state is None if scraping failed

            # --- Display Logs ---
            st.subheader("Combined Scraping Log")


# --- Display Results Area (uses session state) ---
st.subheader("Scraped & Filtered Data")
if st.session_state.combined_df is not None and not st.session_state.combined_df.empty:
    st.dataframe(
        st.session_state.combined_df,
        use_container_width=True,
        column_config={"Media_URL": st.column_config.ImageColumn("Preview Image", width="medium")},
        # row_height = 100 # Increase row height for images if needed
    )
    # Download Button - Placed logically after data display
    @st.cache_data
    def convert_combined_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_combined_df_to_csv(st.session_state.combined_df)
    now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
         label="💾 Download Filtered Data as CSV",
         data=csv_data,
         file_name=f"fb_ads_filtered_scrape_{now_ts}.csv",
         mime='text/csv',
         key='download_csv_button' # Add key for widget uniqueness
    )
elif st.session_state.combined_df is not None and st.session_state.combined_df.empty:
    st.info("Scraping complete, but no ads matched the filtering criteria (Text + Media found).")
else:
    st.info("Click 'Scrape All Terms in Cloud' to fetch data.")


# --- Gemini Processing Button (uses session state) ---
st.subheader(" Analyze Trends (Optional)")
if st.button("Process trends with Gemini?", key='gemini_button', disabled=(GEMINI_API_KEYS is None) ) or ( auto_gemini and st.session_state.combined_df is not None and not st.session_state.combined_df.empty and st.session_state['final_merged_df'] is None) :
    if st.session_state.combined_df is not None and not st.session_state.combined_df.empty:
        df_to_process = st.session_state.combined_df

        # Check if 'Text' column exists
        if "Text" in df_to_process.columns:
            df_to_process  = df_to_process[df_to_process["Text"].str.len() <= 500]

            tokens =count_string_tokens(prompt = "\n".join(list(df_to_process["Text"])),model="gemini-2.0-flash-001	")
            # chunks_num = tokens//60000 + 1
            chunks_num = tokens//140000 + 1
            
            df_appends = []
            max_rows = 3500
            dfs_splits = np.array_split(df_to_process,chunks_num)
            st.text(f"Tokens :{tokens} Num of chucks {len(dfs_splits)}")

          

            for df_idx, df_chunk  in enumerate(dfs_splits):
                st.text(f"{df_idx} {len(df_chunk)}")
                # st.text("\n".join(list(df_chunk["Text"])))
                
                #df_chunk = df_chunk.reset_index(drop=True)
                df_to_process_text  = pd.DataFrame(df_chunk[["Text","Count"]], columns = ["Text","Count"])
                df_to_process_text  = df_to_process_text[df_to_process_text["Text"].str.len() <= 500]
                df_to_process_text['Count'] = pd.to_numeric(df_to_process_text['Count'], errors='coerce')
                df_to_process_text['Count'] = df_to_process_text['Count'].fillna(0)

                # st.text(df_to_process_text)
                df_counts = (
                                df_to_process_text.reset_index()
                                .groupby("Text")
                                .agg(Count=("Count", "sum"), Indices=("index", list))
                                .reset_index()
                            )
                # st.text(df_counts.to_string())
                st.markdown(f"Proccessing {df_idx+1} df...")
                st.dataframe(df_counts)
                
                
                # st.text(df_counts.to_string())
                # st.text("\n".join(list(df_counts)))
                
                
                # --- Prepare prompt (consider limits) ---
                # Example: Use unique texts, limit number of texts sent
             
            
                # # Construct the final prompt
                # gemini_prompt = f"""Please go over the following search arbitrage ideas, deeply think about patterns and reoccurring. I want to get the ideas that would show the most potential. This data is scraped from competitors, so whatever reoccurs is probably successful.\nReturn a list of ideas txt new line delimited!      (no Analysis at all! )of the ideas (just the ideas consicly, no explaning, and not as given), descending order by potential like i described. \nanalyze EACH entry!  BE VERY thorough. be  specific in the topic. don't mix beteern languages and simillar ideas, show them in differnet rows (but still just the ideas consicly , not original input) , return in original language
    
                #         Ad Text:
                #         {'\n'.join(df_to_process["Text"])}
                        
                        
                #         """
                
                df_for_gemini = df_counts["Text"]
                pd.set_option('display.max_colwidth', None)
                df_for_gemini.name ='Index'
                gemini_prompt = """Please go over the following search arbitrage ideas table, deeply think about patterns and reoccurring. I want to get the ideas that would show the most potential. This data is scraped from competitors, so whatever reoccurs is probably successful.\nReturn a list of ideas txt new line delimited!      (no Analysis at all! )of the ideas (just the ideas consicly, no explaning, and not as given), descending order by potential like i described. \nanalyze EACH entry!  BE VERY thorough. be  specific in the topic. don't mix beteern languages (meaning if a topic is in different languages -> different "idea" row ,ALSO  dont mix  simillar but diff topics (New CX-5 ... , Jeep models... are not the same topic!), show them in differnet rows (but still just the ideas consicly , not original input) , return in original language. use the text in 'Text' col to understand the topic and merge simillar text about the similar ideas. then return the indices of the rows from input table per row of output table. return in json example : [{"idea" : "idea text..." , "indices" : [1,50]} , ....]""" + f"""
                I will provide the how many times the text occurred for you and the indices
                Each "idea" value should be 3-6 words include semi specific important keywords
                the idea column texts needs to be a simple concise terms\keyword, no special characters like ( ) & / , etc 
                RETURN ONLY THE JSON NO INTROS OR ANYTHING ELSE!
                table:
                {df_for_gemini.to_string()}"""
            
                st.info(f"Sending  unique text samples to Gemini for analysis...")
                with st.spinner("🧠 Processing with Gemini... This might take a moment."):
                    for trial in range(3):
                        try:
                            with st.expander("Prompt:"):
                                st.text(gemini_prompt)
                            gemini_res = gemini_text_lib(gemini_prompt,model ="gemini-2.5-pro") # Use the dedicated function gemini-2.5-pro-exp-03-25 gemini-2.5-flash
                    
                            if gemini_res:
                                # st.text(gemini_res) 

                                final_df = pd.DataFrame()
                                st.subheader(" Gemini Analysis Results")
                                gemini_res =gemini_res.replace("```json", '').replace("```", '') # Clean up the response
                                #st.text(gemini_res) 
                                # with st.expander("Gemini Results:")::
                                with st.expander("Gemini Completion"):
                                    st.text(gemini_res)
                                gemini_df = pd.read_json(gemini_res) # Convert to DataFrame
                                break
                        except Exception as e:
                            st.text(f"Failed  gemini_df trial :{str(trial)}")
        
    
    
                    for index, row in gemini_df.iterrows():
                        idea = row['idea']
        
                        indices = row['indices']
                        indices = [i for idx in indices for i in df_counts.iloc[idx]['Indices']]
                        # st.text(f"indices : {indices} df_to_process len : {len(df_to_process)} " )

                        # st.text(f"df_counts {df_counts.to_string()}")
                        # st.text(f"indices {str(indices)}")
                        #st.text(f" {idea} indices {indices}")
                        inx_len = sum([df_to_process.loc[idx]["Count"] for idx in indices])
                        #inx_len = len(list(indices))
                        hash_urls={}

                        urls = [df_to_process.loc[idx]["Landing_Page"] for idx in indices]
                        texts = "\n".join(list(set([df_to_process.loc[idx]["Text"] for idx in indices])))
                        
                        # url_title_map = asyncio.run(fetch_all_titles(urls))
                        
                        
                        # count_map ={}

                        # for elem in url_title_map:
                        #    title = elem[1]
                        #    if title not in count_map.keys():
                        #        count_map[title] = 1
                        #    else:
                        #        count_map[title] += 1
                        # max_seen_url_title = max(count_map, key=count_map.get)
                        # max_seen_url =  max_seen_url = next((url for url, title in url_title_map if title == max_seen_url_title),
                        #                                     None)
                        max_seen_url_title = ''
                        # max_seen_url = ''



                        
                        
                        for idx in list(indices): #url:times
                            landing_page = df_to_process.loc[idx]["Landing_Page"]
                            if landing_page in hash_urls:
                                hash_urls[landing_page] += 1
                            else:
                                hash_urls[landing_page] = 1
                        max_seen_url = max(hash_urls, key=hash_urls.get)

                        keys_to_try=['terms','t'] 
                        parsed_url = urlparse(max_seen_url)
                        params = parse_qs(parsed_url.query)
                        domain = str(parsed_url.hostname).replace("www.","")
                        
                        terms = ''.join([val for key in keys_to_try if key in params for val in params[key]])
                        
                        text_urls = {}
                        for idx in list(indices): #text:times
                            text = df_to_process.loc[idx]["Text"]
                            if text in text_urls:
                                text_urls[text] += 1
                            else:
                                text_urls[text] = 1
                        max_seen_text = max(text_urls, key=text_urls.get)
    
    
                        matching_rows = df_to_process.loc[indices]
                        try:
                            if hash_imgs:
                               most_common_hash = get_top_3_media_hashes(matching_rows['Media_URL'].tolist())
                               most_common_img_urls= [elem[1]['data'][0] for elem in most_common_hash]
                               images = "|".join(most_common_img_urls)
                               padded_urls = (list(most_common_img_urls or []) + [None] * 3)[:3]
                            else:
                                images = "|".join(matching_rows['Media_URL'].tolist()[0:2])
                                padded_urls = (matching_rows['Media_URL'].tolist() + [None] * 3)[:3]

                        except Exception as e:
                            print(f"Error processing most_common_img_urls: {e}")


                        

                        
                        img1, img2, img3 = padded_urls


                        try:
                            lang= detect(max_seen_text)
                        except: 
                            lang=''
                        row_df = pd.DataFrame([{
                            "selected" : False,
                            "idea": idea,
                            "lang": lang,
                            "len": inx_len,
                            "max_text": max_seen_text,
                            "max_url": max_seen_url,
                            # "max_seen_url_title" :max_seen_url_title,
                            "terms" : terms,
                            "images": images,
                            "img1": img1,
                            "img2": img2,
                            "img3": img3,
                            "indices": indices, 
                            "domain" : domain,
                            "texts":  texts
                        }])

                        df_appends.append(row_df)

                    else:
                        # Error message already displayed within gemini_text_lib
                        st.error("Gemini processing failed or returned no result.")
        else:
            st.error("Could not find 'Text' column in the scraped data. Cannot analyze.")
    else:
        st.error("No filtered data available to process. Please scrape data first.")

    
    final_merged_df = pd.concat(df_appends)
    # Compact and optimize memory before storing in session
    try:
        final_merged_df = pd.concat(df_appends, ignore_index=True)
    except Exception:
        pass

    before_bytes = 0
    try:
        before_bytes = st.session_state.get('final_merged_df', pd.DataFrame()).memory_usage(deep=True).sum() if isinstance(st.session_state.get('final_merged_df'), pd.DataFrame) else 0
    except Exception:
        pass

    final_merged_df = compact_final_results(final_merged_df)

    try:
        after_bytes = final_merged_df.memory_usage(deep=True).sum()
        if after_bytes:
            st.info(f"Results compacted to {_bytes_to_readable(after_bytes)}")
    except Exception:
        pass

    st.session_state['final_merged_df'] = final_merged_df

    # Release scraped DF to lower memory now that results exist
    st.session_state.combined_df = None

    # Clear cache and force GC
    try:
        st.cache_data.clear()
    except Exception:
        pass
    gc.collect()

elif GEMINI_API_KEYS is None:
    st.warning("Gemini analysis disabled because GEMINI_API_KEY is not configured in secrets.", icon="🚫")



if st.session_state['final_merged_df'] is not None  :
    # df = pd.concat(df_appends)
    # df["selected"] = False  # Ensure column exists
    # st.session_state['final_merged_df'] = df

    # Use a local variable to hold current version
    current_df = st.session_state['final_merged_df'].copy()
 
    # Display controls to reduce UI memory usage
    show_image_previews = st.checkbox("Show image previews", value=False, help="Disable to reduce memory usage on the server and client.")
    total_rows = int(len(current_df))
    safe_max = max(10, total_rows)
    default_rows = min(500, total_rows) if total_rows > 0 else 10
    max_rows_to_display = st.number_input(
        "Max rows to display",
        min_value=10,
        max_value=safe_max,
        value=default_rows,
        step=50,
        help="Showing fewer rows reduces memory usage.")

    # Build lightweight column configuration
    link_col = getattr(st.column_config, 'LinkColumn', st.column_config.TextColumn)
    col_cfg = {
        "idea": st.column_config.TextColumn(pinned=True),
        "selected": st.column_config.CheckboxColumn("Selected", pinned=True),
    }
    if show_image_previews:
        col_cfg.update({
            'img1': st.column_config.ImageColumn("Image 1", width="small"),
            'img2': st.column_config.ImageColumn("Image 2", width="small"),
            'img3': st.column_config.ImageColumn("Image 3", width="small"),
        })
    else:
        col_cfg.update({
            'img1': link_col("Image 1 URL"),
            'img2': link_col("Image 2 URL"),
            'img3': link_col("Image 3 URL"),
        })

    # Display editor on a limited slice - do NOT bind directly to session_state
    edited_df = st.data_editor(
        current_df.head(max(0, int(max_rows_to_display))),
        column_config=col_cfg,
        use_container_width=True,
        hide_index=True,
    )

    # Memory management actions
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Free memory now", help="Drop intermediates, clear cache, and GC."):
            try:
                st.session_state['final_merged_df_selected'] = None
            except Exception:
                pass
            st.session_state.combined_df = None
            try:
                st.cache_data.clear()
            except Exception:
                pass
            gc.collect()

    # Let user manually confirm selection changes to sync
    is_gen_html = st.checkbox("Gen HTML content")
    if st.button("Process Selected Rows"):
        st.session_state['final_merged_df_selected'] = edited_df.copy()

        # Work with updated session state
        selected_df = st.session_state['final_merged_df_selected'][st.session_state['final_merged_df_selected']["selected"] == True]

        if is_gen_html:
            title_res = []
            html_res=[]
            for index, row in selected_df.iterrows():
                tries = 0
                done = False
                while tries < 5 and done is False:
                        
                    try:

                        content = get_html_content(row['max_url'])
                        # st.text(content)
                        prompt = """write as html using only  <a>, <p>, <h2>–<h4>, <li>, <ul>, <img>.\nNEVER use <br> or <br\> or <ol> or <ol\> NEVER!
                        only the article content no footers no images!! no images! no writer name!, no <div>!!! first element is ALWAYS <p>. NEVER write\return the domain name ( like xxx.com) in the title or html , omit that!! return in language same as input . return json dict, 2 keys : 'title', 'html'  . \n example :{"title" : "Learn more about how veterans ...", 'html' :"full article w/o title with html tags..'}  no <div>\n\n""" + content
                        gemini_res =gemini_text_lib(prompt=prompt, model='gemini-2.0-flash-exp' ) # gemini-2.0-flash-exp
                        # st.text(gemini_res)
                        pure_html = gemini_res.replace("```html","").replace("```","").replace("```json","").replace("json","")
                        pure_html = json.loads(pure_html)
                        done = True
    
                    except Exception as e:
                        pure_html = f"error {e} "
                        tries += 1
                    # st.text(str(pure_html)) 
                    title_res.append(pure_html['title'].replace("```json",""))
                    html_res.append(pure_html['html'].replace("```html","").replace("```","").replace("```json",""))

            selected_df['html'] = html_res
            selected_df['html_title'] = title_res

                
    # if st.button("👁 Show Selected Rows"):
        st.dataframe(selected_df, hide_index=True, use_container_width=True,column_config={
            'img1': st.column_config.ImageColumn("Image 1", width="medium"),
            'img2': st.column_config.ImageColumn("Image 2", width="medium"),
            'img3': st.column_config.ImageColumn("Image 3", width="medium")})

# --- Footer ---
st.markdown("---")
