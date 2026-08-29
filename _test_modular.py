from RMFCalculator.eos.finite_nucleus import compute_Pb208, compute_Sn132

r = compute_Pb208("FSUGold")
print(f"Binding/A = {r['E_per_nucleon']:.4f} MeV")
print(f"r_charge  = {r['r_charge']:.4f} fm")
print(f"converged = {r['converged']}")

r2 = compute_Sn132("FSUGold")
print(f"Sn132 Binding/A = {r2['E_per_nucleon']:.4f} MeV")