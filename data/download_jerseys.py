"""
Jersey Image Downloader – WM 2026
Zwei Queries pro Nation (Home + Away), temp-Ordner verhindert Dateiname-Kollisionen.
"""

import shutil
from icrawler.builtin import BingImageCrawler
from pathlib import Path

BASE_DIR = Path(__file__).parent / "jerseys"
TEMP_DIR = Path(__file__).parent / "_temp_download"

NATIONS = {
    "argentina":   ["argentina home football jersey 2024", "argentina away football jersey 2024"],
    "australia":   ["australia socceroos home jersey 2024", "australia socceroos away jersey 2024"],
    "belgium":     ["belgium home football jersey 2024", "belgium away football jersey 2024"],
    "brazil":      ["brazil home football jersey 2024", "brazil away football jersey 2024"],
    "croatia":     ["croatia home football jersey 2024", "croatia away football jersey 2024"],
    "england":     ["england home football jersey 2024", "england away football jersey 2024"],
    "france":      ["france home football jersey 2024", "france away football jersey 2024"],
    "germany":     ["germany home football jersey 2024", "germany away football jersey 2024"],
    "japan":       ["japan home football jersey 2024", "japan away football jersey 2024"],
    "netherlands": ["netherlands home football jersey 2024", "netherlands away football jersey 2024"],
    "portugal":    ["portugal home football jersey 2024", "portugal away football jersey 2024"],
    "southafrica": ["south africa bafana home jersey 2024", "south africa bafana away jersey 2024"],
    "spain":       ["spain home football jersey 2024", "spain away football jersey 2024"],
    "switzerland": ["switzerland home football jersey 2024", "switzerland away football jersey 2024"],
    "usa":         ["usa usmnt home jersey 2024", "usa usmnt away jersey 2024"],
}

TARGET = 25


def count_images(folder: Path) -> int:
    return sum(1 for f in folder.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def download_to_temp(query: str, max_num: int) -> list[Path]:
    """Lädt Bilder in temp-Ordner und gibt die Pfade zurück."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    crawler = BingImageCrawler(storage={"root_dir": str(TEMP_DIR)})
    crawler.crawl(keyword=query, max_num=max_num)
    return list(TEMP_DIR.glob("*"))


def download_nation(nation: str, queries: list[str]):
    folder = BASE_DIR / nation
    folder.mkdir(parents=True, exist_ok=True)

    existing = count_images(folder)
    if existing >= TARGET:
        print(f"  {nation}: bereits {existing} Bilder – übersprungen")
        return

    for query in queries:
        current = count_images(folder)
        if current >= TARGET:
            break
        needed = TARGET - current
        print(f"   '{query}' (brauche noch {needed})...")

        temp_files = download_to_temp(query, max_num=needed + 5)
        copied = 0
        for src in temp_files:
            if count_images(folder) >= TARGET:
                break
            # Eindeutiger Dateiname: nation_001.jpg, nation_002.jpg ...
            idx = count_images(folder) + 1
            dst = folder / f"{nation}_{idx:03d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            copied += 1
        print(f"   → {copied} neue Bilder (gesamt: {count_images(folder)})")

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)


if __name__ == "__main__":
    print("Jersey Download startet...\n")
    for nation, queries in NATIONS.items():
        print(f"\n→ {nation}")
        download_nation(nation, queries)

    print("\n✓ Fertig! Bilder pro Nation:")
    for nation in NATIONS:
        count = count_images(BASE_DIR / nation)
        mark = "✓" if count >= TARGET else f"⚠  nur {count}"
        print(f"  {nation:15s}: {count}  {mark}")
