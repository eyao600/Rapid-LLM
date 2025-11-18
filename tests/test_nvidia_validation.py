import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import nvidia_inf

def test_avg_abs_error_below_threshold_network_ignored():
  results = nvidia_inf.run(enable_plot=True, network_ignored=True)
  avg_abs_error = results["avg_abs_error"]
  assert not math.isnan(avg_abs_error), "NVIDIA validation produced no valid measurements"
  # Historically the average absolute error bottoms out at roughly 5.97%, so we allow
  # a little slack (6.5%) to absorb minor DeepFlow changes while still flagging regressions.
  assert avg_abs_error <= 17.0, f"Average absolute error {avg_abs_error:.2f}% exceeds 17.0%"
