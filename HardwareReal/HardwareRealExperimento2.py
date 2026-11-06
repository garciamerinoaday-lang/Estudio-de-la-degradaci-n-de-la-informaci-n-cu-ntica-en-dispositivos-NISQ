"""
Experimento 2
Ejecución en hardware real - Usando ibm_fez
Aday García Merino
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np
import time


def extract_counts(pub_result):
    data = pub_result.data
    if hasattr(data, 'get_counts'):
        return data.get_counts()
    for attr_name in dir(data):
        if attr_name.startswith('_'):
            continue
        attr = getattr(data, attr_name)
        if hasattr(attr, 'get_counts'):
            return attr.get_counts()
    if hasattr(data, 'keys'):
        for key in data.keys():
            if hasattr(data[key], 'get_counts'):
                return data[key].get_counts()
    raise AttributeError("No se encontró un objeto con método get_counts en data")


# Bloque robusto
def identity_block_robust():
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


# Configuración
service = QiskitRuntimeService()
backend_name = "ibm_fez"
backend = service.backend(backend_name)
print(f"Backend: {backend.name}")

depths = [0, 1, 2, 3, 5, 7, 10, 15, 20]
n_runs = 3  # Número de repeticiones
all_fidelities = []

# Para cada ejecución
for run in range(n_runs):
    print(f"\n=== Ejecución {run + 1}/{n_runs} ===")
    circuits = []
    for d in depths:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        for _ in range(d):
            qc.compose(identity_block_robust(), inplace=True)
        qc.cx(0, 1)
        qc.h(0)
        qc.measure([0, 1], [0, 1])
        circuits.append(qc)

    # Transpilar con optimization_level=0
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    circs_transpiled = [pm.run(qc) for qc in circuits]

    # Verificar profundidades
    print("Profundidades reales:")
    for i, d in enumerate(depths):
        print(f"  Depth {d:2d} -> real depth {circs_transpiled[i].depth():4d}")

    # Sampler con mitigación (si funciona)
    try:
        sampler = Sampler(backend, options={"resilience_level": 1})
    except:
        sampler = Sampler(backend)

    # Ejecutar
    job = sampler.run(circs_transpiled, shots=8192)
    print(f"Job ID: {job.job_id()}")
    result = job.result()

    # Extraer fidelidades
    fids = []
    for i, d in enumerate(depths):
        counts = extract_counts(result[i])
        prob_00 = counts.get('00', 0) / sum(counts.values())
        fid = np.sqrt(prob_00)
        fids.append(fid)
        print(f"Depth {d:2d}: P(00)={prob_00:.4f}, F={fid:.4f}")
    all_fidelities.append(fids)

    # Pequeña pausa entre ejecuciones para no saturar
    time.sleep(5)

# Promediar resultados
avg_fidelities = np.mean(all_fidelities, axis=0)
std_fidelities = np.std(all_fidelities, axis=0)

print("\n=== Resultados promediados ===")
for i, d in enumerate(depths):
    print(f"Depth {d:2d}: F = {avg_fidelities[i]:.4f} ± {std_fidelities[i]:.4f}")

# Graficar
plt.figure(figsize=(10, 6))
plt.errorbar(depths, avg_fidelities, yerr=std_fidelities, fmt='o-', capsize=5)
plt.xlabel('Número de bloques identidad')
plt.ylabel('Fidelidad')
plt.title(f'Decaimiento de la fidelidad (promedio de {n_runs} ejecuciones)\nBackend: {backend.name}')
plt.grid(True, alpha=0.3)
plt.savefig('exp2_robusto_promediado.png', dpi=150)
plt.show()
print("Gráfica guardada.")