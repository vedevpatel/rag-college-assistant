import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import os
from pypdf import PdfReader
from io import BytesIO
from tqdm import tqdm # Progress bar

# --- Configuration ---
SEED_PAGES = [
    # Your list of seed URLs from the original post...
    "https://www.laspositascollege.edu/", # Add the homepage as a good starting point
    "https://www.laspositascollege.edu/president/index.php",
    "https://www.laspositascollege.edu/prospective/index.php",
    "https://www.laspositascollege.edu/gv/index.php",
    "https://www.laspositascollege.edu/research/index.php",
    "http://www.lpcfoundation.org/",
    "https://www.laspositascollege.edu/about/index.php",
    "https://www.laspositascollege.edu/adminservices/index.php",
    "https://www.laspositascollege.edu/childcenter/index.php",
    "https://www.laspositascollege.edu/outreach/index.php",
    "https://www.laspositascollege.edu/educationalpartnerships/index.php",
    "https://www.tocite.net/laspositascollege/portal", # May require login - might fail
    "https://www.laspositascollege.edu/about/consumerinfo.php",
    "https://www.laspositascollege.edu/measure-a/index.php",
    "http://lpcexpressnews.com/",
    "https://havikjournal.wixsite.com/website", # External, might have different structure
    "https://www.laspositascollege.edu/admissions/index.php",
    "https://www.opencccapply.net/cccapply-welcome?cccMisCode=481", # External application portal
    "https://www.laspositascollege.edu/admissions/concurrent.php",
    "https://www.laspositascollege.edu/international/index.php",
    "https://www.laspositascollege.edu/admissions/forms.php", # Check for PDF links here
    "https://www.laspositascollege.edu/admissions/transcripts.php",
    "https://www.laspositascollege.edu/admissions/fees.php",
    "https://www.laspositascollege.edu/admissions/requirements.php",
    "https://www.laspositascollege.edu/admissions/registration.php",
    "https://www.laspositascollege.edu/financialaid/index.php",
    "https://www.laspositascollege.edu/financialaid/applications.php",
    "https://www.laspositascollege.edu/financialaid/forms-2024-2025.php", # Check for PDF links here
    "https://www.laspositascollege.edu/financialaid/eligibility.php",
    "https://lpc.financialaidtv.com/", # External video site
    "https://www.laspositascollege.edu/financialaid/waiver.php",
    "https://www.laspositascollege.edu/financialaid/scholarshipinfo.php",
    "https://www.laspositascollege.edu/financialaid/aid.php",
    "https://www.laspositascollege.edu/counseling/index.php",
    "https://www.laspositascollege.edu/counseling/forms.php", # Check for PDF links here
    "https://www.laspositascollege.edu/stepstosuccess/preparecounselingappointment.php",
    "https://www.laspositascollege.edu/counseling/priority.php",
    "https://www.laspositascollege.edu/counseling/courses.php",
    "https://www.laspositascollege.edu/studentservices/index.php",
    "https://www.laspositascollege.edu/assessment/index.php",
    "https://www.laspositascollege.edu/bcrc/index.php",
    "https://www.laspositascollege.edu/careercenter/index.php",
    "https://www.laspositascollege.edu/dsps/index.php",
    "https://www.laspositascollege.edu/healthcenter/index.php",
    "https://www.laspositascollege.edu/transfercenter/index.php",
    "https://www.laspositascollege.edu/veterans/index.php",
    "https://www.laspositascollege.edu/tutorialcenter/index.php",
    "https://www.laspositascollege.edu/raw/index.php",
    "https://www.laspositascollege.edu/computercenter/index.php",
    "http://www.bkstr.com/laspositasstore/home", # External Bookstore
    "https://www.laspositascollege.edu/ztc/index.php",
    "https://www.laspositascollege.edu/studentlife/index.php",
    "https://www.laspositascollege.edu/stepstosuccess/newstudent.php",
    "https://www.laspositascollege.edu/lpcsg/index.php",
    "https://www.laspositascollege.edu/clubs/index.php",
    "https://www.laspositascollege.edu/icc/agendas.php", # Likely PDFs
    "https://www.laspositascollege.edu/basicneeds/index.php",
    "https://www.laspositascollege.edu/academicservices/index.php",
    "https://www.laspositascollege.edu/programs/index.php",
    "https://www.laspositascollege.edu/academicintegrity/index.php",
    "https://www.laspositascollege.edu/lpcarticulation/index.php",
    "https://www.laspositascollege.edu/communityed/index.php",
    "https://www.laspositascollege.edu/educationalpartnerships/earlycollegecredit.php",
    "https://www.laspositascollege.edu/gpas/index.php",
    "https://www.laspositascollege.edu/catalog/current/programs/",
    "https://www.laspositascollege.edu/admissions/academic-calendar.php", # Check for PDF links
    "https://www.laspositascollege.edu/class-schedule/index.php", # Check for PDF links
    "https://laspositascollege.edu/catalog/", # Check for PDF links (main catalog)
    "https://www.laspositascollege.edu/class-schedule/finals.php",
    "https://www.laspositascollege.edu/industrycredentials/index.php",
    "https://www.laspositascollege.edu/apprenticeship/index.php",
    "https://www.laspositascollege.edu/programs/noncredit.php",
    "https://www.laspositascollege.edu/cpl/index.php",
    "https://www.laspositascollege.edu/performingarts/index.php",
    "https://athletics.laspositascollege.edu/landing/index", # Simpler athletics link
    "https://www.laspositascollege.edu/campushillwinery/index.php"
]
ALLOWED_DOMAINS = {
    "www.laspositascollege.edu",
    "laspositascollege.edu",
    "lpcexpressnews.com",
    "www.lpcfoundation.org",
    "lpcfoundation.org",
    "athletics.laspositascollege.edu"
    # Add other relevant subdomains if needed
    # Be careful with external domains like bkstr, wixsite, financialaidtv - they might have very different structures or block scraping.
    # Consider if you *really* need info from opencccapply or tocite, as they are portals.
}
OUTPUT_FILE = "scraped_data.jsonl"
MAX_DEPTH = 2 # Limit crawl depth to avoid excessive requests
REQUEST_DELAY = 0.5 # Seconds to wait between requests to be polite
REQUEST_TIMEOUT = 15 # Seconds to wait for a response

# --- Helper Functions ---

def is_valid_url(url):
    """Check if the URL is well-formed and within allowed domains."""
    try:
        parsed = urlparse(url)
        # Basic scheme check and ensure it's http or https
        if parsed.scheme not in ['http', 'https']:
            return False
        # Check if domain is allowed
        if parsed.netloc not in ALLOWED_DOMAINS:
            # Allow paths relative to allowed domains (e.g. /admissions/ on main site)
            # This logic relies on urljoin handling relative paths correctly later
            return parsed.netloc == "" and parsed.path != ""
        return True
    except Exception:
        return False

def extract_text_from_html(content, url):
    """Extract meaningful text from HTML content."""
    soup = BeautifulSoup(content, 'html.parser')
    page_title = soup.title.string.strip() if soup.title else url

    # Remove common noise elements more aggressively
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button', 'iframe', 'img', 'meta', 'link']):
        tag.decompose()

    # Attempt to find the main content area (common IDs/classes, may need adjustment)
    main_content = soup.find('main') or \
                   soup.find(id='main-content') or \
                   soup.find(class_='main-content') or \
                   soup.find(id='content') or \
                   soup.find(class_='content') or \
                   soup.body # Fallback to body if no main structure found

    if not main_content:
        main_content = soup.body # Ensure we have something

    # Get text, trying to preserve some structure with separators
    text = main_content.get_text(separator='\n', strip=True)

    # Basic cleaning: remove excessive newlines
    cleaned_text = '\n'.join(line for line in text.splitlines() if line.strip())
    return page_title, cleaned_text

def extract_text_from_pdf(content, url):
    """Extract text from PDF content."""
    text = ""
    page_title = os.path.basename(urlparse(url).path) # Use filename as title
    try:
        with BytesIO(content) as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        cleaned_text = '\n'.join(line for line in text.splitlines() if line.strip())
        return page_title, cleaned_text
    except Exception as e:
        print(f"[WARN] Failed to extract text from PDF {url}: {e}")
        return page_title, "" # Return empty string on failure

def crawl_and_scrape(start_url, visited_urls, output_f, depth=1):
    """Recursive crawler and scraper."""
    if depth > MAX_DEPTH or start_url in visited_urls:
        return set() # Return empty set for new links

    print(f"{'  ' * (depth-1)}[Depth {depth}] Visiting: {start_url}")
    visited_urls.add(start_url)
    time.sleep(REQUEST_DELAY) # Polite delay

    try:
        headers = {'User-Agent': 'LPC-RAG-Bot/1.0 (Python requests; +https://www.laspositascollege.edu/)'} # Identify your bot
        response = requests.get(start_url, timeout=REQUEST_TIMEOUT, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        content_type = response.headers.get('content-type', '').lower()
        content = response.content
        page_title = start_url
        extracted_text = ""
        internal_links = set()

        if 'html' in content_type:
            page_title, extracted_text = extract_text_from_html(content, start_url)
            # Extract links only from HTML
            soup = BeautifulSoup(content, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:') or href.startswith('javascript:'):
                    continue

                full_url = urljoin(start_url, href)
                # Clean URL fragment
                parsed_link = urlparse(full_url)
                cleaned_url = parsed_link._replace(fragment="").geturl()

                if is_valid_url(cleaned_url) and cleaned_url not in visited_urls:
                     # Check domain again after resolving relative paths
                    link_domain = urlparse(cleaned_url).netloc
                    if link_domain in ALLOWED_DOMAINS:
                        internal_links.add(cleaned_url)

        elif 'pdf' in content_type:
            page_title, extracted_text = extract_text_from_pdf(content, start_url)
            # No links to extract from PDFs themselves in this simple approach

        else:
            print(f"[WARN] Skipping unsupported content type '{content_type}' at {start_url}")
            return set() # No links to follow from non-HTML/PDF

        # Save extracted content if it's meaningful
        if extracted_text and len(extracted_text.split()) > 10: # Basic check for meaningful content length
            data = {"source": start_url, "page_title": page_title, "content": extracted_text}
            output_f.write(json.dumps(data) + '\n')
            print(f"{'  ' * (depth-1)}  -> Saved content from: {start_url} (Title: {page_title})")
        else:
            print(f"{'  ' * (depth-1)}  -> No meaningful content found/extracted from: {start_url}")

        # Recurse
        newly_found_links = set()
        for link in internal_links:
            newly_found_links.update(crawl_and_scrape(link, visited_urls, output_f, depth + 1))

        return internal_links.union(newly_found_links) # Return all links found at this level and below

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network/HTTP error for {start_url}: {e}")
        return set()
    except Exception as e:
        print(f"[ERROR] Processing error for {start_url}: {e}")
        return set()

# --- Main Execution ---
if __name__ == "__main__":
    visited = set()
    all_found_links = set()

    # Ensure seed pages are valid and add them
    initial_urls = {url for url in SEED_PAGES if is_valid_url(url)}

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            print(f"Starting crawl from {len(initial_urls)} seed URLs. Max depth: {MAX_DEPTH}. Output: {OUTPUT_FILE}")
            # Using tqdm for progress over seed URLs
            for seed_url in tqdm(initial_urls, desc="Crawling Seeds"):
                if seed_url not in visited:
                     all_found_links.update(crawl_and_scrape(seed_url, visited, f, depth=1))
                else:
                    print(f"Skipping already visited seed: {seed_url}")

        print(f"\nCrawling complete. Visited {len(visited)} unique URLs.")
        print(f"Total unique internal links found (approx): {len(all_found_links)}")
        print(f"Scraped data saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred during the main crawl loop: {e}")
