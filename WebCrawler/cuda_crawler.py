import requests
from bs4 import BeautifulSoup
import csv
import re
import logging
from urllib.parse import urljoin
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class CUDADocCrawler:
    def __init__(self):
        self.base_url = "https://docs.nvidia.com/cuda/"
        self.visited_urls = set()
        self.data = []

    def crawl(self):
        main_page = self.fetch_page(self.base_url)
        if main_page:
            links = self.extract_links(main_page)
            for link in links:
                self.process_page(link)
                time.sleep(1)

    def fetch_page(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def extract_links(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith(('http://', 'https://')):
                full_url = href
            else:
                full_url = urljoin(self.base_url, href)
            if full_url.startswith(self.base_url) and full_url.endswith('.html'):
                links.append({"title": a.get_text().strip(), "url": full_url})
        return links

    def process_page(self, link):
        url = link['url']
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        logger.info(f"Processing: {url}")

        html = self.fetch_page(url)
        if html:
            content = self.extract_content(html)
            cleaned_content = self.clean_content(content)
            section = self.get_section(url)

            self.data.append({
                "section": section,
                "title": link['title'],
                "url": url,
                "content": cleaned_content
            })

    def extract_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        main_content = soup.find('div', class_='body')
        if not main_content:
            main_content = soup.find('main') or soup.find('article') or soup.find('section')
        
        if main_content:
            for nav in main_content.find_all(['div', 'ul'], class_=['nav', 'toc']):
                nav.decompose()
            return main_content.get_text(separator=' ', strip=True)
        return ""

    def clean_content(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
        words = text.split()[:1000]
        return ' '.join(words)

    def get_section(self, url):
        sections = {
            "installation-guide": "Installation Guide",
            "programming-guide": "Programming Guide",
            "cuda-runtime-api": "CUDA Runtime API",
            "cuda-driver-api": "CUDA Driver API",
            "cuda-math-api": "CUDA Math API",
            "cuda-toolkit": "CUDA Toolkit",
            "cuda-compiler-driver-nvcc": "NVCC",
            "cuda-memcheck": "CUDA Memcheck",
            "cuda-gdb": "CUDA GDB",
            "cusparse": "cuSPARSE",
            "cublas": "cuBLAS",
            "cufft": "cuFFT",
            "curand": "cuRAND",
            "thrust": "Thrust",
        }
        for key, value in sections.items():
            if key in url:
                return value
        return "Miscellaneous"

    def save_data(self, csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["section", "title", "url", "content"])
            writer.writeheader()
            writer.writerows(self.data)

if __name__ == "__main__":
    crawler = CUDADocCrawler()
    crawler.crawl()
    crawler.save_data("cuda_document.csv")
    logger.info(f"Crawling complete. Data saved to cuda_documentation.csv")
