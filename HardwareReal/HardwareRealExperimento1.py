"""
Experimento 1: Fidelidad de un estado de Bell mediante eco de Loschmidt
Ejecución en hardware real - Usando ibm_fez
Aday García Merino
"""

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np

def extract_counts(pub_result):
    """Extrae counts del resultado de un Sampler."""
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

# =============================================
# 1. Configuración del backend (ibm_fez)
# =============================================
service = QiskitRuntimeService()

# Especificamos explícitamente ibm_fez (mismo que en Experimento 2)
backend_name = "ibm_fez"
backend = service.backend(backend_name)

# Verificar que el backend está operativo
print("=" * 60)
print(f" Backend seleccionado: {backend.name}")
print(f"   - Número de qubits: {backend.num_qubits}")
print(f"   - ¿Operacional? {backend.status().operational}")
print("=" * 60)

# =============================================
# 2. Construcción del circuito de eco de Loschmidt
# =============================================
# Circuito: preparación de Bell + eco (inverso) + medición
qc = QuantumCircuit(2, 2)

# Preparación del estado de Bell |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
qc.h(0)
qc.cx(0, 1)

# Barrera para separar visualmente preparación del eco
qc.barrier()

# Eco (inverso de la preparación) - idealmente recupera |00⟩
qc.cx(0, 1)
qc.h(0)

# Medición
qc.measure([0, 1], [0, 1])

print("\nCircuito cuántico (eco de Loschmidt):")
print(qc.draw(output='text'))

# =============================================
# 3. Transpilación
# =============================================
# Usamos optimization_level=1 (optimización moderada)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
qc_transpiled = pm.run(qc)

print(f"\nProfundidad del circuito transpilado: {qc_transpiled.depth()}")
print(f"Número de puertas: {qc_transpiled.size()}")

# =============================================
# 4. Ejecución en hardware
# =============================================
sampler = Sampler(backend)

# Ejecutamos con 10000 shots (mismo que en el Experimento 2)
shots = 10000
job = sampler.run([qc_transpiled], shots=shots)
print(f"\n Job ID: {job.job_id()}")
print(" Esperando resultados...")

# Obtener resultados
result = job.result()
pub_result = result[0]

# Extraer counts
counts = extract_counts(pub_result)

# =============================================
# 5. Cálculo de la fidelidad
# =============================================
total_shots = sum(counts.values())
prob_00 = counts.get('00', 0) / total_shots
fidelity = np.sqrt(prob_00)

print("\n" + "=" * 60)
print(" RESULTADOS")
print("=" * 60)
print(f"Distribución de resultados: {counts}")
print(f"\nProbabilidad |00⟩: {prob_00:.4f}")
print(f"Fidelidad: {fidelity:.4f}")

# Mostrar también otras probabilidades para referencia
print("\nDistribución completa:")
for state in ['00', '01', '10', '11']:
    prob = counts.get(state, 0) / total_shots
    print(f"  {state}: {prob:.4f} ({counts.get(state, 0)} counts)")

print("=" * 60)