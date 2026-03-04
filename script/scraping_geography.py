import json
import os
import re

import requests
import rootutils
from bs4 import BeautifulSoup
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def get_from_url(url, params=None):
    try:
        HEADERS = {"User-Agent": "my-wikipedia-script/0.1 (you@example.com)"}
        r = requests.get(url, params=params, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Error for url: {}".format(url))
        return None


def extract_topics(geo_id: str):
    search_response = get_from_url(f"https://data.census.gov/api/profile/metadata?g={geo_id}&includeHighlights=false")
    if search_response == None: return None
    topics = []
    for section in search_response['sections']:
        topics.append(section['title'])
        topics.extend(sub['topic'] for sub in section['subsections'] if sub['type'] == "topic")
    topics = set(topics)
    if "Explore Your State" in topics: topics.remove("Explore Your State")
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


def extract_US(path: str):
    entities_response = get_from_url("https://data.census.gov/api/search?g=010XX00US&services=entities")['response']
    assert entities_response != None
    for state in tqdm(entities_response['profiles'], desc='Processing US states'):

        page = state['title']

        # ======================== CENSUS ========================
        geo_id = state['facets']['geography'][0]
        description_response = get_from_url(
            F"https://data.census.gov/api/profile/metadata?g={geo_id}&includeHighlights=false")
        if description_response == None: continue
        geography = {
            'title': page,
            'level': state['level'],
            'geoID': geo_id,
            'short description': clean_text(state['teaser']) if 'teaser' in state.keys() else "",
            'description': clean_text(description_response['header']['description'])
        }

        topics = extract_topics(geo_id)
        if topics == None: continue

        for topic in topics:
            response = get_from_url(f"https://data.census.gov/api/profile/content/topic?g={geo_id}&infoSection={topic}")
            if response == None: continue
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
        sections = get_from_url("https://en.wikipedia.org/w/api.php",
                                params={
                                    "action": "parse",
                                    "page": page,
                                    "prop": "sections",
                                    "format": "json",
                                    "formatversion": "2",
                                }
                                )["parse"]["sections"]
        if sections == None: continue
        #  Retrieve paragraphs
        for section in sections:
            response = get_from_url("https://en.wikipedia.org/w/api.php",
                                    params={
                                        "action": "parse",
                                        "page": page,
                                        "prop": "text",
                                        "section": section['index'],
                                        "format": "json",
                                        "formatversion": "2",
                                    }
                                    )["parse"]["text"]
            if response == None: continue
            soup = BeautifulSoup(response, "html.parser")
            for par in soup.select("sup.reference"):
                par.decompose()
            paragraphs = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)])
            geography.update({
                f"wiki_{section['anchor']}": clean_text(paragraphs)
            })

        filename = os.path.join(path, re.sub(r"\s+", "_", geography["title"]) + ".json")
        with open(filename, "w") as f:
            json.dump(geography, f, indent=5)


def extract_NER_entites(path: str):
    gpe_entities = json.load(open("note/topic_results.json"))['PLACES']
    for entity in gpe_entities:

        # ======================== WIKIPEDIA ========================
        #  Retrieve sections
        try:
            sections = get_from_url("https://en.wikipedia.org/w/api.php",
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
            response = get_from_url("https://en.wikipedia.org/w/api.php",
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

        filename = os.path.join(path, re.sub(r"\s+", "_", geography["title"]) + ".json")
        with open(filename, "w") as f:
            json.dump(geography, f, indent=5)


if __name__ == '__main__':
    path = "knowledge_base/geography/"
    os.makedirs(path, exist_ok=True)
    extract_US(path)
    extract_NER_entites(path)
