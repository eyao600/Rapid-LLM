import math
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import nvidia_inf

# Test for H100 device configuration needs more refinement
# NOT WORKING YET

@pytest.mark.xfail(reason="Pending model refinements", strict=False)
def test_avg_abs_error_below_threshold_network_ignored():
  results = nvidia_inf.run(enable_plot=False, network_ignored=True, device="H100")
  avg_abs_error = results["avg_abs_error"]
  assert not math.isnan(avg_abs_error), "IMEC validation produced no valid measurements"
  # Historically the average absolute error bottoms out at roughly 6.5%, so we allow
  # a little slack (6.5%) to absorb minor DeepFlow changes while still flagging regressions.
  assert avg_abs_error <= 6.5, f"Average absolute error {avg_abs_error:.2f}% exceeds 6.5%"

  # TODO: split these into tests per device/model

if __name__ == "__main__":
  test_avg_abs_error_below_threshold_network_ignored()
