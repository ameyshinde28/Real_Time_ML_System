import pytest
from datetime import datetime, timezone
from src.utils import timestamp_ms_to_human_readable_utc

def test_timestamp_ms_to_human_readable_utc():
    # Test case 1: Known timestamp
    timestamp_ms = 1609459200000  # 2021-01-01 00:00:00 UTC
    expected_output = "2021-01-01 00:00:00 UTC"
    assert timestamp_ms_to_human_readable_utc(timestamp_ms) == expected_output

    # Test case 2: Current time
    current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_output = timestamp_ms_to_human_readable_utc(current_ms)
    assert current_output.endswith("UTC")
    
    # Test case 3: Leap year
    leap_year_ms = 1582934400000  # 2020-02-29 00:00:00 UTC
    expected_leap_output = "2020-02-29 00:00:00 UTC"
    assert timestamp_ms_to_human_readable_utc(leap_year_ms) == expected_leap_output