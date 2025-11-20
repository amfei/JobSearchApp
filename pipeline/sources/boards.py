"""
boards.py —  job scraper with optional Selenium for detailed descriptions.

This module works locally (e.g., VSCode) using a full Chrome driver, 
and automatically disables Selenium when running on Streamlit Cloud 
or in environments without a graphical interface.
"""

import os
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Try to use undetected-chromedriver when available (for anti-bot protection)
try:
    import undetected_chromedriver as uc
    HAS_UC = True
except Exception:
    HAS_UC = False



# Utility Functions

def safe_sleep(a: float = 0.8, b: float = 2.0):
    """Pause randomly between 'a' and 'b' seconds to reduce the risk of being rate-limited."""
    time.sleep(random.uniform(a, b))


def setup_chromedriver_options():
    """Configure Chrome options for headless execution."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if os.path.exists("/usr/bin/chromium-browser"):
        options.binary_location = "/usr/bin/chromium-browser"
    return options


def get_driver():
    """Initialize and return a Chrome driver (uses undetected-chromedriver if available)."""
    if HAS_UC:
        opts = uc.ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        if os.path.exists("/usr/bin/chromium-browser"):
            opts.binary_location = "/usr/bin/chromium-browser"
        return uc.Chrome(options=opts)
    else:
        return webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=setup_chromedriver_options()
        )


def extract_linkedin_description(job_url: str, use_selenium: bool = True) -> str:
    """Extract the full job description text from a LinkedIn job page using Selenium."""
    if not use_selenium:
        return ""

    try:
        driver = get_driver()
        driver.get(job_url)
        safe_sleep(0.8, 1.8)
        desc = ""
        try:
            el = driver.find_element(By.XPATH, "//section[contains(@class, 'description')]")
            desc = el.text.strip()
        except Exception:
            pass
        finally:
            driver.quit()
        return " ".join(desc.split()) if desc else ""
    except Exception:
        return ""



# LinkedIn Job Scraping

def fetch_linkedin_jobs(
    job_titles: List[str],
    locations: List[str],
    days_filter: int = 14,
    num_jobs: int = 100,
    excluded_titles: List[str] = None,
    use_selenium: bool = False
) -> List[Dict[str, Any]]:
    """
    Scrape job postings from public LinkedIn search results.

    Parameters
    ----------
    job_titles : List[str]
        List of job titles to search for.
    locations : List[str]
        List of target locations.
    days_filter : int
        Maximum age of job postings (in days).
    num_jobs : int
        Maximum number of jobs to retrieve per title-location pair.
    excluded_titles : List[str]
        Titles to ignore (e.g., Junior, Manager).
    use_selenium : bool
        If True, fetch detailed job descriptions with Selenium.

    Returns
    -------
    List[Dict[str, Any]]
        A list of job dictionaries with title, company, location, link, and optional description.
    """
    if excluded_titles is None:
        excluded_titles = [
            "Data Engineer", "Junior", "Intern", "Officer", "Head", "Director", "Manager"
        ]

    headers = {"User-Agent": "Mozilla/5.0"}
    all_jobs: List[Dict[str, Any]] = []

    for location in locations:
        print(f"Searching jobs in {location}")
        for title in job_titles:
            query = title.replace(" ", "%20")
            url = f"https://www.linkedin.com/jobs/search?keywords={query}&location={location}"
            print(f"   {title}: {url}")

            try:
                response = requests.get(url, headers=headers, timeout=20)
                if response.status_code != 200:
                    print(f"Request failed with status {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                cards = soup.find_all("div", class_="base-card")[:num_jobs]

                for card in cards:
                    title_elem = card.find("h3", class_="base-search-card__title")
                    company_elem = card.find("h4", class_="base-search-card__subtitle")
                    loc_elem = card.find("span", class_="job-search-card__location")
                    link_elem = card.find("a", class_="base-card__full-link") or card.find("a")
                    date_elem = card.find("time")

                    if not (title_elem and company_elem and loc_elem and link_elem):
                        continue

                    title_text = title_elem.get_text(strip=True)
                    company_text = company_elem.get_text(strip=True)
                    loc_text = loc_elem.get_text(strip=True)
                    link = (link_elem.get("href") or "").strip()

                    # Skip excluded titles
                    if any(bad.lower() in title_text.lower() for bad in excluded_titles):
                        continue

                    # Apply date filter
                    if date_elem and date_elem.has_attr("datetime"):
                        try:
                            post_date = datetime.strptime(date_elem["datetime"], "%Y-%m-%d")
                            if datetime.now() - post_date > timedelta(days=days_filter):
                                continue
                        except Exception:
                            pass

                    # Optionally fetch job description
                    desc = extract_linkedin_description(link, use_selenium=use_selenium)

                    job_info = {
                        "title": title_text,
                        "company": company_text,
                        "location": loc_text,
                        "link": link,
                        "description": desc,
                    }
                    all_jobs.append(job_info)
                    safe_sleep(0.4, 1.2)

            except Exception as e:
                print(f"Error fetching {title} in {location}: {e}")
                continue

    print(f"Total scraped jobs: {len(all_jobs)}")
    return all_jobs
