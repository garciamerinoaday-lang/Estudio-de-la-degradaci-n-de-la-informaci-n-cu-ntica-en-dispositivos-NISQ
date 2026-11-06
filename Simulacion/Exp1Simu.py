
"""
Experimento 1: Fidelidad de un estado de Bell mediante eco de Loschmidt
Simulación
Aday García Merino
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, amplitude_damping_error,
                              phase_damping_error, depolarizing_error,
                              ReadoutError)
from qiskit.visualization import plot_histogram

# =============================================
# 1. Parámetros de ruido (modelo corregido)
# =============================================
# Tiempos de puerta (valores típicos)
GATE_TIME_1Q = 100e-9    # 100 ns
GATE_TIME_CX = 560e-9    # 560 ns

# Parámetros del backend (ibm_fez) - mismos para ambos qubits
T1 = 187.2e-6    # 187.2 µs
T2 = 251.3e-6    # 251.3 µs

# Calcular Tphi (desfase puro) a partir de T1 y T2
# 1/T2 = 1/(2*T1) + 1/Tphi  =>  Tphi = 1 / (1/T2 - 1/(2*T1))
Tphi = 1 / (1/T2 - 1/(2*T1))

# Probabilidades de error para puertas de 1 qubit
gamma_ad = 1 - np.exp(-GATE_TIME_1Q / T1)      # amplitude damping
gamma_phi = 1 - np.exp(-GATE_TIME_1Q / Tphi)   # phase damping puro (NO T2)

# Error depolarizante en CNOT (1%)
cx_depolarizing = 0.01

# Error de lectura (promedio)
readout_error = 0.03192
readout_matrix = [[1 - readout_error, readout_error],
                  [readout_error, 1 - readout_error]]

# Error depolarizante adicional en puertas de 1 qubit (0.1%)
depol_1q_extra = 0.001

print("=" * 70)
print("📊 MODELO DE RUIDO CORREGIDO (con Tphi)")
print("=" * 70)
print(f"Tiempo puerta 1Q: {GATE_TIME_1Q * 1e9:.0f} ns")
print(f"Tiempo puerta CX: {GATE_TIME_CX * 1e9:.0f} ns")
print(f"T1 = {T1 * 1e6:.1f} µs, T2 = {T2 * 1e6:.1f} µs (ambos qubits)")
print(f"Tphi = {Tphi * 1e6:.1f} µs  (desfase puro)")
print(f"γ_ad = {gamma_ad:.6f}")
print(f"γ_phi = {gamma_phi:.6f}  (phase damping puro)")
print(f"Error depolarizante CX: {cx_depolarizing*100:.1f}%")
print(f"Error depolarizante extra 1Q: {depol_1q_extra*100:.1f}%")
print(f"Readout error: {readout_error:.5f} (matriz simétrica)")
print("=" * 70)

# =============================================
# 2. Circuito de eco de Loschmidt
# =============================================
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.barrier()
qc.cx(0, 1)
qc.h(0)
qc.measure([0, 1], [0, 1])

print("\nCircuito cuántico (eco de Loschmidt):")
print(qc.draw(output='text'))

# =============================================
# 3. Simulación ideal (sin ruido)
# =============================================
print("\n" + "=" * 70)
print("--- SIMULACIÓN IDEAL (sin ruido) ---")
print("=" * 70)

sim_ideal = AerSimulator()
t_qc = transpile(qc, sim_ideal)
job_ideal = sim_ideal.run(t_qc, shots=10000)
counts_ideal = job_ideal.result().get_counts(qc)

total_ideal = sum(counts_ideal.values())
prob_00_ideal = counts_ideal.get('00', 0) / total_ideal
fidelity_ideal = np.sqrt(prob_00_ideal)

print("Resultados:")
for state in ['00', '01', '10', '11']:
    prob = counts_ideal.get(state, 0) / total_ideal * 100
    print(f"  {state} : {prob:.2f}%")
print(f"\nProbabilidad |00⟩ = {prob_00_ideal:.4f}")
print(f"Fidelidad = {fidelity_ideal:.4f}")

# =============================================
# 4. Construcción del NoiseModel con modelo corregido
# =============================================
noise_model = NoiseModel()

# Errores de amplitude damping (con T1) y phase damping puro (con Tphi)
error_ad = amplitude_damping_error(gamma_ad)
error_phi = phase_damping_error(gamma_phi)   # ¡USAMOS Tphi, NO T2!

# Error depolarizante adicional para puertas de 1 qubit (0.1%)
error_depol_1q = depolarizing_error(depol_1q_extra, 1)

# Puertas de 1 qubit en el circuito
one_qubit_gates = ['h', 'x']

for gate in one_qubit_gates:
    # Aplicar a qubit 0
    noise_model.add_quantum_error(error_ad, gate, [0])
    noise_model.add_quantum_error(error_phi, gate, [0])   # phase damping puro
    noise_model.add_quantum_error(error_depol_1q, gate, [0])
    # Aplicar a qubit 1
    noise_model.add_quantum_error(error_ad, gate, [1])
    noise_model.add_quantum_error(error_phi, gate, [1])   # phase damping puro
    noise_model.add_quantum_error(error_depol_1q, gate, [1])

# Error depolarizante en CNOT (1%)
error_cx_dep = depolarizing_error(cx_depolarizing, 2)
noise_model.add_quantum_error(error_cx_dep, 'cx', [0, 1])

# Errores de lectura (misma matriz para ambos qubits)
noise_model.add_readout_error(readout_matrix, [0])
noise_model.add_readout_error(readout_matrix, [1])

print("\n✅ Modelo de ruido configurado (con Tphi corregido)")

# =============================================
# 5. Simulación con ruido
# =============================================
print("\n" + "=" * 70)
print("--- SIMULACIÓN CON RUIDO (modelo corregido) ---")
print("=" * 70)

sim_noisy = AerSimulator(noise_model=noise_model)
t_qc_noisy = transpile(qc, sim_noisy)
job_noisy = sim_noisy.run(t_qc_noisy, shots=10000)
counts_noisy = job_noisy.result().get_counts(qc)

total_noisy = sum(counts_noisy.values())
prob_00_noisy = counts_noisy.get('00', 0) / total_noisy
fidelity_noisy = np.sqrt(prob_00_noisy)

print("Resultados:")
for state in ['00', '01', '10', '11']:
    prob = counts_noisy.get(state, 0) / total_noisy * 100
    print(f"  {state} : {prob:.2f}%")
print(f"\nProbabilidad |00⟩ = {prob_00_noisy:.4f}")
print(f"Fidelidad = {fidelity_noisy:.4f}")


print("\nSimulación:")
print(f"  Probabilidad |00⟩ = {prob_00_noisy:.4f}")
print(f"  Fidelidad = {fidelity_noisy:.4f}")
print(f"  Distribución: {counts_noisy}")
