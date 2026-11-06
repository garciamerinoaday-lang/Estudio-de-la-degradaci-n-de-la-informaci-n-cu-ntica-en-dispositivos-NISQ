"""
Experimento 2 - Simulación con ruido real + depolarizing 0.1% en puertas 1Q
Aday García Merino
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, amplitude_damping_error,
                              phase_damping_error, ReadoutError,
                              depolarizing_error)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# =============================================
# 1. Parámetros del experimento
# =============================================
shots = 8192
depths = [0, 1, 2, 3, 5, 7, 10, 15, 20]
n_runs = 3

# =============================================
# 2. Parámetros de ruido reales
# =============================================
GATE_TIME_1Q = 100e-9      # 100 ns
GATE_TIME_CX = 560e-9      # 560 ns

# Datos del backend (ibm_fez)
T1 = 187.2e-6   # 187.2 µs
T2 = 251.3e-6   # 251.3 µs

# Calcular Tphi (desfase puro) a partir de T1 y T2
# 1/T2 = 1/(2*T1) + 1/Tphi  =>  Tphi = 1 / (1/T2 - 1/(2*T1))
Tphi = 1 / (1/T2 - 1/(2*T1))

# Probabilidades de error para puertas de 1 qubit
gamma_ad = 1 - np.exp(-GATE_TIME_1Q / T1)      # amplitude damping
gamma_phi = 1 - np.exp(-GATE_TIME_1Q / Tphi)   # phase damping puro (NO T2)

# Error depolarizante en CNOT (1%)
cx_depolarizing = 0.01

# Error de lectura promedio
readout_error = 0.03192
readout_matrix = [[1 - readout_error, readout_error],
                  [readout_error, 1 - readout_error]]

# Error depolarizante adicional para puertas de 1 qubit (0.1%)
depol_1q_extra = 0.001

print("=" * 70)
print("MODELO DE RUIDO CORREGIDO (con Tphi) - Experimento 2")
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
# 3. Bloque identidad (nominal, sin error coherente)
# =============================================
def identity_block():
    qc = QuantumCircuit(2)
    theta = 0.1
    phi = 0.2
    qc.rx(theta, 0)
    qc.ry(phi, 0)
    qc.barrier()
    qc.rx(theta, 1)
    qc.ry(phi, 1)
    qc.barrier()
    qc.ry(-phi, 1)
    qc.rx(-theta, 1)
    qc.barrier()
    qc.ry(-phi, 0)
    qc.rx(-theta, 0)
    qc.barrier()
    return qc

# =============================================
# 4. Construcción de circuitos
# =============================================
def build_circuits(depths):
    circuits = []
    for d in depths:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        for _ in range(d):
            qc.compose(identity_block(), inplace=True)
        qc.cx(0, 1)
        qc.h(0)
        qc.measure([0, 1], [0, 1])
        circuits.append(qc)
    return circuits

circuits = build_circuits(depths)

# =============================================
# 5. Modelo de ruido (con depolarizing extra 0.1% en 1Q)
# =============================================
def create_noise_model():
    noise_model = NoiseModel()

    # Errores de amplitude damping (con T1) y phase damping puro (con Tphi)
    error_ad = amplitude_damping_error(gamma_ad)
    error_phi = phase_damping_error(gamma_phi)   # ¡USAMOS Tphi, NO T2!

    # Errores depolarizantes adicionales en 1Q (0.1%)
    error_depol_1q = depolarizing_error(depol_1q_extra, 1)

    one_qubit_gates = ['rx', 'ry', 'h', 'x']
    for gate in one_qubit_gates:
        # Añadir amplitude damping y phase damping puro a ambos qubits
        noise_model.add_quantum_error(error_ad, gate, [0])
        noise_model.add_quantum_error(error_phi, gate, [0])   # phase damping puro
        noise_model.add_quantum_error(error_ad, gate, [1])
        noise_model.add_quantum_error(error_phi, gate, [1])   # phase damping puro

        # Añadir depolarizing extra (0.1%) a ambos qubits
        noise_model.add_quantum_error(error_depol_1q, gate, [0])
        noise_model.add_quantum_error(error_depol_1q, gate, [1])

    # Error depolarizante en CNOT (1%)
    error_cx_dep = depolarizing_error(cx_depolarizing, 2)
    noise_model.add_quantum_error(error_cx_dep, 'cx', [0, 1])

    # Errores de lectura (matriz simétrica para ambos qubits)
    noise_model.add_readout_error(readout_matrix, [0])
    noise_model.add_readout_error(readout_matrix, [1])

    return noise_model

# =============================================
# 6. Simulación
# =============================================
print("\n" + "="*70)
print("Simulación: Modelo con ruido real + depolarizing 0.1% en puertas 1Q")
print("="*70)

noise_model = create_noise_model()
sim = AerSimulator(noise_model=noise_model)

# Transpilación (sin coupling map, asumiendo conectividad completa)
pm = generate_preset_pass_manager(backend=sim, optimization_level=0)
circs_transpiled = [pm.run(qc) for qc in circuits]

print("\nProfundidades reales después de transpilar:")
for i, d in enumerate(depths):
    print(f"  Depth {d:2d} -> real depth {circs_transpiled[i].depth():4d}")

all_fidelities = []
all_probabilities = []

for run in range(n_runs):
    print(f"\n  Ejecución {run+1}/{n_runs}")
    fids = []
    probs = []
    for i, qc in enumerate(circs_transpiled):
        job = sim.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        prob_00 = counts.get('00', 0) / shots
        fid = np.sqrt(prob_00)
        fids.append(fid)
        probs.append(prob_00)
        print(f"    Depth {depths[i]:2d}: P(00)={prob_00:.4f}, F={fid:.4f}")
    all_fidelities.append(fids)
    all_probabilities.append(probs)
    time.sleep(0.5)

# Promedios
avg_fids = np.mean(all_fidelities, axis=0)
std_fids = np.std(all_fidelities, axis=0)
avg_probs = np.mean(all_probabilities, axis=0)
std_probs = np.std(all_probabilities, axis=0)

print("\n  Resultados promediados:")
print(f"{'Depth':<8} {'P(00)':<12} {'Fidelidad':<12}")
for i, d in enumerate(depths):
    print(f"{d:<8} {avg_probs[i]:.4f} ± {std_probs[i]:.4f}   {avg_fids[i]:.4f} ± {std_fids[i]:.4f}")

print("\n Simulación completada.")