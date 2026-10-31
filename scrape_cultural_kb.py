"""Scrapes country-scoped cultural facts from English Wikipedia into cultural_data.json.

Pages that do not exist are reported at the end - check that list and fix titles so every
country ends up with a healthy number of facts (aim for 100+ per country).
"""
import hashlib
import json
import re
import time

import wikipediaapi

import config

# Put a real contact address here (Wikipedia asks for an identifying User-Agent).
wiki = wikipediaapi.Wikipedia(user_agent="CulturalKBPipeline/2.0 (student project; contact@example.com)",
                              language="en")

COUNTRY_PAGES = {
    "South Korea": ["Culture of South Korea", "Korean cuisine", "Public holidays in South Korea",
                    "Chuseok", "Seollal", "Hanbok", "Kimchi", "Korean tea", "Education in South Korea",
                    "Sport in South Korea", "Korean wedding", "Religion in South Korea",
                    "Music of Korea", "Korean folklore", "Korean royal court cuisine"],
    "Ethiopia": ["Culture of Ethiopia", "Ethiopian cuisine", "Public holidays in Ethiopia", "Timkat",
                 "Meskel", "Enkutatash", "Injera", "Ethiopian coffee ceremony", "Education in Ethiopia",
                 "Music of Ethiopia", "Ethiopian calendar", "Sport in Ethiopia", "Religion in Ethiopia",
                 "Ethiopian wedding", "Ethiopian Orthodox Tewahedo Church"],
    "China": ["Culture of China", "Chinese cuisine", "Public holidays in China", "Chinese New Year",
              "Mid-Autumn Festival", "Dragon Boat Festival", "Qingming Festival", "Education in China",
              "Sport in China", "Chinese tea culture", "Chinese wedding", "Religion in China",
              "Music of China", "Chinese martial arts", "Lantern Festival"],
    "Indonesia": ["Culture of Indonesia", "Indonesian cuisine", "Public holidays in Indonesia", "Nyepi",
                  "Batik", "Education in Indonesia", "Music of Indonesia", "Sport in Indonesia",
                  "Religion in Indonesia", "Gamelan", "Wayang", "Indonesian wedding",
                  "Eid al-Fitr in Indonesia", "Rendang", "Nasi goreng"],
    "Azerbaijan": ["Culture of Azerbaijan", "Azerbaijani cuisine", "Public holidays in Azerbaijan",
                   "Novruz", "Education in Azerbaijan", "Music of Azerbaijan", "Sport in Azerbaijan",
                   "Religion in Azerbaijan", "Azerbaijani carpet", "Mugham", "Plov",
                   "Azerbaijani wedding", "Tea in Azerbaijan", "Dolma", "Azerbaijani dance"],
}

SKIP_SECTIONS = {"see also", "references", "external links", "further reading", "notes",
                 "bibliography", "sources", "footnotes", "gallery"}


def chunk_text(text, max_chars=600, min_chars=80):
    """Pack whole sentences into chunks of at most ~max_chars (no mid-sentence truncation)."""
    chunks, current = [], ""
    for paragraph in (p.strip() for p in re.split(r"\n+", text) if p.strip()):
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if len(current) + len(sentence) + 1 <= max_chars:
                current = f"{current} {sentence}".strip()
            else:
                if len(current) >= min_chars:
                    chunks.append(current)
                current = sentence[: max_chars * 2]
    if len(current) >= min_chars:
        chunks.append(current)
    return chunks


def walk_sections(sections):
    for section in sections:
        if section.title.lower() in SKIP_SECTIONS:
            continue
        yield section
        yield from walk_sections(section.sections)


def extract_facts_from_page(page, country, language):
    facts = []
    blocks = [("Overview", page.summary)] + [(s.title, s.text) for s in walk_sections(page.sections)]
    for category, text in blocks:
        for chunk in chunk_text(text):
            facts.append({"country": country, "language": language, "category": category,
                          "topic": page.title, "fact": chunk, "source": page.fullurl})
    return facts


def main():
    all_facts, seen, missing = [], set(), []
    for country, titles in COUNTRY_PAGES.items():
        language = config.COUNTRY_LANGUAGE[country]
        print(f"Fetching data for {country}...")
        country_count = 0
        for title in titles:
            page = wiki.page(title)
            time.sleep(0.2)
            if not page.exists():
                missing.append(f"{country}: {title}")
                continue
            new = 0
            for fact in extract_facts_from_page(page, country, language):
                digest = hashlib.md5(fact["fact"].lower().encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                all_facts.append(fact)
                new += 1
            country_count += new
            print(f"  - {title}: {new} facts")
        print(f"  => {country}: {country_count} facts")

    with open(config.DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_facts, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_facts)} facts to {config.DATA_PATH}")
    if missing:
        print("\nPages not found (fix the titles in COUNTRY_PAGES):")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
