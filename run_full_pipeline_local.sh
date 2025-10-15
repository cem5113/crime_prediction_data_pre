#!/usr/bin/env bash
set -euo pipefail

python -u update_crime.py
python -u update_911.py
python -u update_311.py
python -u update_population.py
python -u update_bus.py
python -u update_train.py
python -u update_poi.py           || python -u scripts/pipeline_make_sf_crime_06.py
python -u update_police_gov.py    || python -u scripts/enrich_police_gov_06_to_07.py
python -u scripts/update_weather.py

echo "DONE → $(ls -lh crime_data/sf_crime_08.csv 2>/dev/null || ls -lh sf_crime_08.csv)"
