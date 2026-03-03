import json
import re

import requests
import rootutils
from bs4 import BeautifulSoup
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def GET(url, params=None):
    HEADERS = {"User-Agent": "my-wikipedia-script/0.1 (you@example.com)"}
    r = requests.get(url, params=params, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def extract_topics():
    search_response = GET("https://data.census.gov/api/profile/metadata?g=010XX00US&includeHighlights=false")
    topics = []
    for section in search_response['sections']:
        topics.append(section['title'])
        topics.extend(sub['topic'] for sub in section['subsections'] if sub['type'] == "topic")
    topics = set(topics)
    topics.remove("Explore Your State")
    return topics


def clean_text(content):
    """Cleans text content by removing unwanted characters and formatting issues."""
    content = str(content)
    # Remove HTML tags
    content = BeautifulSoup(content, "html.parser").get_text()
    # Normalize whitespace characters to single spaces
    content = re.sub(r'[\t\n\r\f\v]+', ' ', content)
    # Collapse multiple spaces into one
    content = re.sub(r' {2,}', ' ', content)
    return content.strip()


def extract_US():
    topics = extract_topics()
    entities_response = GET("https://data.census.gov/api/search?g=010XX00US&services=entities")['response']
    for state in tqdm(entities_response['profiles'], desc='Processing US states'):

        page = state['title']

        # ======================== CENSUS ========================
        geography = {}
        geo_id = state['facets']['geography'][0]
        description_response = GET(F"https://data.census.gov/api/profile/metadata?g={geo_id}&includeHighlights=false")
        geography = {
            'title': page,
            'level': state['level'],
            'geoID': geo_id,
            'short description': clean_text(state['teaser']) if 'teaser' in state.keys() else "",
            'description': clean_text(description_response['header']['description'])
        }

        for topic in topics:
            response = GET(f"https://data.census.gov/api/profile/content/topic?g={geo_id}&infoSection={topic}")
            if response['vizData'] != {} and len(response['vizData']['stats']) > 0:
                geography.update({
                    topic: {
                        'title': response['vizData']['title'],
                        'source': response['vizData']['source'],
                        'stats': response['vizData']['stats'],

                    }
                })

        # ======================== WIKIPEDIA ========================
        #  Retrieve sections
        sections = GET("https://en.wikipedia.org/w/api.php",
                       params={
                           "action": "parse",
                           "page": page,
                           "prop": "sections",
                           "format": "json",
                           "formatversion": "2",
                       }
                       )["parse"]["sections"]
        #  Retrieve paragraphs
        for section in sections:
            response = GET("https://en.wikipedia.org/w/api.php",
                           params={
                               "action": "parse",
                               "page": page,
                               "prop": "text",
                               "section": section['index'],
                               "format": "json",
                               "formatversion": "2",
                           }
                           )["parse"]["text"]
            soup = BeautifulSoup(response, "html.parser")
            for par in soup.select("sup.reference"):
                par.decompose()
            paragraphs = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)])
            geography.update({
                f"wiki_{section['anchor']}": clean_text(paragraphs)
            })

        with open(f"knowledge_base/geography/{geography['title']}.json", "w") as f:
            json.dump(geography, f, indent=5)


def extract_NER_entites():
    gpe_entities = json.load(open("note/topic_results.json"))['PLACES']
    for entity in gpe_entities:

        # ======================== WIKIPEDIA ========================
        #  Retrieve sections
        try:
            sections = GET("https://en.wikipedia.org/w/api.php",
                           params={
                               "action": "parse",
                               "page": entity,
                               "prop": "sections",
                               "format": "json",
                               "formatversion": "2",
                           }
                           )["parse"]["sections"]
        except:
            continue

        geography = {
            'title': entity,
            'source': 'Wikipedia',
        }
        #  Retrieve paragraphs
        for section in sections:
            if section in {"External_links", "Further_reading", "Sources", "Notes", "See_also", "Media"}: continue
            response = GET("https://en.wikipedia.org/w/api.php",
                           params={
                               "action": "parse",
                               "page": entity,
                               "prop": "text",
                               "section": section['index'],
                               "format": "json",
                               "formatversion": "2",
                           }
                           )["parse"]["text"]
            soup = BeautifulSoup(response, "html.parser")
            for par in soup.select("sup.reference"):
                par.decompose()
            paragraphs = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)])
            geography.update({
                f"wiki_{section['anchor']}": clean_text(paragraphs)
            })

        with open(f"knowledge_base/geography/{geography['title']}.json", "w") as f:
            json.dump(geography, f, indent=5)


if __name__ == '__main__':
    extract_US()
    extract_NER_entites()
