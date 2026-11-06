# Estudio-de-la-degradaci-n-de-la-informaci-n-cu-ntica-en-dispositivos-NISQ
This Bachelor’s Thesis investigates the degradation of quantum information in NISQ (Noisy
Intermediate-Scale Quantum) devices through two experiments performed on a real superconducting processor and in simulation. Using the Loschmidt echo technique on a Bell state, the
state fidelity is quantified after circuits of varying depth. The first experiment, with a simple circuit, shows excellent agreement between real hardware and simulations based on standard noise
models (T1 relaxation, T2 dephasing, depolarizing channel, and readout error). However, in the
second experiment, where identity sequences are interleaved to accumulate errors, the simulation systematically overestimates fidelity, with differences up to 0.27 for the greatest depth. The
main cause is identified as the omission of the actual processor topology in the simulation, which
underestimates the effective circuit depth (by a factor of up to 5). Furthermore, the order of
gates significantly affects error accumulation, indicating the need to incorporate coherent errors
(such as overrotations) into the models. The conclusions highlight the importance of including
hardware architecture and coherent effects to predict fidelities in deep circuits, and point the
way toward more realistic noise models and tailored mitigation strategies.

