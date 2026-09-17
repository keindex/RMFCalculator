"""Plot four meson fields and the charge density of Ca-48."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RMFCalculator.eos.finite_nucleus.params import FSU_DELTA67_FN
from RMFCalculator.eos.finite_nucleus.solver import compute_finite_nucleus


result = compute_finite_nucleus(Z=20, N=28, params=FSU_DELTA67_FN)

fields = [
    ("sigma", r"$\sigma(r)$", "#1769aa"),
    ("omega", r"$\omega(r)$", "#d55e00"),
    ("rho03", r"$\rho_0^3(r)$", "#009e73"),
    ("delta", r"$\delta(r)$", "#cc79a7"),
]

fig, axis = plt.subplots(figsize=(9, 6))
for field_name, label, color in fields:
    axis.plot(
        result["r"],
        result[field_name],
        color=color,
        linewidth=2,
        label=label,
    )

neutron_density = result["n_v"] - result["n_gamma"]
density_axis = axis.twinx()
proton_line, = density_axis.plot(
    result["r"],
    result["n_gamma"],
    color="#e69f00",
    linewidth=2,
    linestyle="--",
    label=r"$n_p(r)$",
)
neutron_line, = density_axis.plot(
    result["r"],
    neutron_density,
    color="#56b4e9",
    linewidth=2,
    linestyle="-.",
    label=r"$n_n(r)$",
)
radius_line = axis.axvline(
    result["r_charge"],
    color="#000000",
    linewidth=1.5,
    linestyle=":",
    label=rf"$r_{{\rm ch}}={result['r_charge']:.2f}$ fm",
)

axis.set_xlabel(r"$r$ (fm)")
axis.set_ylabel(r"Field (fm$^{-1}$)")
density_axis.set_ylabel(r"Nucleon density (fm$^{-3}$)", color="#555555")
density_axis.tick_params(axis="y", labelcolor="#555555")
axis.set_title(r"$^{48}$Ca: meson fields and nucleon densities")
axis.grid(True, alpha=0.3)
field_handles, field_labels = axis.get_legend_handles_labels()
axis.legend(
    field_handles[:-1] + [proton_line, neutron_line, radius_line],
    field_labels[:-1]
    + [proton_line.get_label(), neutron_line.get_label(), radius_line.get_label()],
    loc="upper right",
)
fig.tight_layout()

output_path = PROJECT_ROOT / "img" / "ca48_four_fields.png"
fig.savefig(output_path, dpi=180)
print(f"Saved: {output_path}")
print(f"Charge rms radius: {result['r_charge']:.6f} fm")
