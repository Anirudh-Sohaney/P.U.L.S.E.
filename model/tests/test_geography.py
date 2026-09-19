import io
import json

import pandas as pd

from arkansas_pharma_signal import geography


class _Response:
    def __init__(self, body):
        self.body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_census_county_weather_points_parse_all_arkansas_counties(monkeypatch):
    rows = ["USPS|GEOID|GEOIDFQ|ANSICODE|NAME|ALAND|AWATER|ALAND_SQMI|AWATER_SQMI|INTPTLAT|INTPTLONG"]
    for index in range(75):
        rows.append(f"AR|{5001 + index:05d}|fq|ansi|County {index}|0|0|0|0|34.{index:02d}|-92.{index:02d}")
    monkeypatch.setattr(geography, "urlopen", lambda request, timeout: _Response("\n".join(rows)))
    result, metadata = geography.fetch_census_county_weather_points(url="https://example.test")
    assert len(result) == 75
    assert result["county_fips"].is_unique
    assert metadata["county_count"] == 75
    assert set(result["arkansas_region"]) <= {"northwest", "northeast", "central", "southwest", "southeast", ""}
