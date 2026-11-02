"""Run local-data examples in a temporary directory, without downloads or GUI."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import uproot

ROOT = Path(__file__).resolve().parents[2]


def check_lhcb_units(work, env):
    """Check rest and opposite-momentum pairs using the CERN branch layout."""
    data = {f"H{i}_P{axis}": np.zeros(2) for i in range(1, 4) for axis in "XYZ"}
    data["H1_PX"][1] = 300.  # MeV/c
    data["H2_PX"][1] = -300.
    with uproot.recreate(work / "B2HHH_MagnetDown.root") as file:
        tree = file.mktree("DecayTree", {name: "float64" for name in data})
        tree.extend(data)
    code = """
import runpy
import numpy as np
example = runpy.run_path('08_lhcb_data.py')
m12, m23, m13 = example['analyze_data']('B2HHH_MagnetDown.root')
mass = 0.13957
np.testing.assert_allclose(m12, [2 * mass, 2 * np.hypot(0.3, mass)], rtol=1e-12)
pair_mass = np.sqrt(2 * mass**2 + 2 * mass * np.hypot(0.3, mass))
for result in [m23, m13]:
    np.testing.assert_allclose(result, [2 * mass, pair_mass], rtol=1e-12)
"""
    subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code],
                   cwd=work, env=env, check=True, timeout=120)


def main():
    with tempfile.TemporaryDirectory(prefix="lvec-examples-") as directory:
        work = Path(directory)
        for path in (ROOT / "examples").glob("*.py"):
            shutil.copy2(path, work / path.name)
        (work / "samples").mkdir()
        (work / "plots").mkdir()
        env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(work / "matplotlib"))
        scripts = [work / "create_test_data.py", *sorted(work.glob("[0-9]*.py"))]
        for script in scripts:
            if script.name == "08_lhcb_data.py":
                check_lhcb_units(work, env)
            result = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", str(script)],
                                    cwd=work, env=env, capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f"{script.name}\n{result.stdout}\n{result.stderr}")
            print(f"PASS {script.name}", flush=True)


if __name__ == "__main__":
    main()
