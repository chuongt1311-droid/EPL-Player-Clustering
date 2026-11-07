"""Rebuilds docs/index.html from cache/payload.json (written by the notebook), so page changes do not need a notebook rerun. Usage: python tools/rebuild_page.py"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from build_site import build_site
build_site(json.load(open(ROOT / 'cache' / 'payload.json', encoding='utf-8')), ROOT / 'docs' / 'index.html'); print('rebuilt docs/index.html')
