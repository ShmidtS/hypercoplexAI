#!/usr/bin/env python
"""
Генератор датасета v6 — 300+ пар кроссдоменных аналогий.
Добавляет новые семейства: квантмех, статфизика, нейронные сети, экономика, астрофизика, биохимия.
"""
import json
from pathlib import Path

# Загружаем v5 как базу
v5 = json.loads(Path('data/real_pairs_v5.json').read_text(encoding='utf-8'))
max_group = max(x['group_id'] for x in v5)

# Домены: 0=physics/engineering, 1=biology/medicine, 2=computer_science/math, 3=chemistry/materials
# Для v6 добавляем новые пары в тех же доменах

new_pairs = [
    # ====== КВАНТОВАЯ МЕХАНИКА ↔ ТЕРМОДИНАМИКА ======
    {
        "source_text": "Quantum tunneling allows particles to pass through classically forbidden energy barriers, with exponentially decreasing probability.",
        "source_domain": 0,
        "target_text": "Thermal activation enables molecules to overcome energy barriers via Arrhenius mechanism, with exponential temperature dependence.",
        "target_domain": 1,
        "relation": "positive",
        "group_id": max_group + 1,
        "family": "barrier_tunneling_activation"
    },
    {
        "source_text": "Wave function collapse during measurement forces quantum superposition into a definite classical state.",
        "source_domain": 0,
        "target_text": "Phase transition collapses continuous symmetry into a specific ordered ground state below critical temperature.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 2,
        "family": "symmetry_breaking_collapse"
    },
    {
        "source_text": "Quantum entanglement creates non-local correlations between particles that persist regardless of distance.",
        "source_domain": 0,
        "target_text": "Allosteric regulation creates long-range correlations between distant protein sites through conformational propagation.",
        "target_domain": 1,
        "relation": "positive",
        "group_id": max_group + 3,
        "family": "nonlocal_correlation_propagation"
    },
    {
        "source_text": "Heisenberg uncertainty principle prevents simultaneous precise measurement of position and momentum.",
        "source_domain": 0,
        "target_text": "Bias-variance tradeoff prevents simultaneous minimization of model error and generalization error in machine learning.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 4,
        "family": "uncertainty_tradeoff_principle"
    },
    {
        "source_text": "Quantum decoherence destroys superposition through interaction with environment, leading to classical behavior.",
        "source_domain": 0,
        "target_text": "Catastrophic forgetting destroys previously learned knowledge through overwriting during new task learning in neural networks.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 5,
        "family": "decoherence_forgetting"
    },
    # ====== НЕЙРОННЫЕ СЕТИ ↔ СТАТИСТИЧЕСКАЯ ФИЗИКА ======
    {
        "source_text": "Backpropagation computes gradients by propagating error signals backwards through the network layers.",
        "source_domain": 2,
        "target_text": "Monte Carlo simulation propagates random perturbations through the system to sample equilibrium configurations.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 6,
        "family": "gradient_propagation_sampling"
    },
    {
        "source_text": "Attention mechanism computes weighted importance of all input positions for each output position.",
        "source_domain": 2,
        "target_text": "Mean field theory approximates each particle's environment as the average field produced by all other particles.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 7,
        "family": "attention_mean_field"
    },
    {
        "source_text": "Dropout regularization randomly removes neurons during training to prevent co-adaptation and overfitting.",
        "source_domain": 2,
        "target_text": "Thermal fluctuations randomly perturb molecular configurations, preventing system from getting trapped in local energy minima.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 8,
        "family": "stochastic_regularization_fluctuation"
    },
    {
        "source_text": "Batch normalization standardizes layer inputs by removing mean and scaling by variance for training stability.",
        "source_domain": 2,
        "target_text": "Renormalization group rescales physical quantities at different length scales to extract universal behavior near critical points.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 9,
        "family": "normalization_renormalization"
    },
    {
        "source_text": "Residual connections in deep networks allow gradients to flow directly through skip connections without vanishing.",
        "source_domain": 2,
        "target_text": "Bypass grafts in cardiovascular surgery create alternative flow paths around blocked vessels to restore circulation.",
        "target_domain": 1,
        "relation": "positive",
        "group_id": max_group + 10,
        "family": "bypass_shortcut_flow"
    },
    # ====== ЭКОНОМИКА ↔ ФИЗИКА ======
    {
        "source_text": "Brownian motion of stock prices follows random walk with drift, driven by continuous arrival of new information.",
        "source_domain": 2,
        "target_text": "Brownian motion of pollen particles follows random walk driven by continuous thermal collisions with fluid molecules.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 11,
        "family": "brownian_random_walk"
    },
    {
        "source_text": "Market bubbles form when positive feedback loops drive asset prices far above fundamental values before collapse.",
        "source_domain": 2,
        "target_text": "Plasma instabilities form when positive feedback amplifies perturbations beyond equilibrium before disruption.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 12,
        "family": "bubble_instability_collapse"
    },
    {
        "source_text": "Supply and demand equilibrium determines price where quantity supplied equals quantity demanded.",
        "source_domain": 2,
        "target_text": "Chemical equilibrium determines concentration where forward and reverse reaction rates are equal.",
        "target_domain": 3,
        "relation": "positive",
        "group_id": max_group + 13,
        "family": "equilibrium_balance_point"
    },
    {
        "source_text": "Portfolio diversification reduces risk by combining uncorrelated assets that partially cancel each other's fluctuations.",
        "source_domain": 2,
        "target_text": "Error correction codes reduce noise by combining redundant signals that partially cancel each other's errors.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 14,
        "family": "diversification_redundancy"
    },
    {
        "source_text": "Arbitrage exploits price differences between markets by simultaneously buying cheap and selling expensive.",
        "source_domain": 2,
        "target_text": "Osmosis exploits concentration differences by moving solvent from low to high solute concentration through membrane.",
        "target_domain": 3,
        "relation": "positive",
        "group_id": max_group + 15,
        "family": "gradient_exploitation_flow"
    },
    # ====== БИОЛОГИЯ ↔ ИНФОРМАЦИОННЫЕ ТЕХНОЛОГИИ ======
    {
        "source_text": "DNA replication uses template strand to create identical copy with extremely low error rate through proofreading.",
        "source_domain": 1,
        "target_text": "RAID storage replicates data across multiple drives to create redundant copies with error detection and correction.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 16,
        "family": "replication_redundancy_storage"
    },
    {
        "source_text": "Gene regulatory networks control protein expression through transcription factor binding and feedback loops.",
        "source_domain": 1,
        "target_text": "Neural network weights control signal propagation through learned connections and activation functions.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 17,
        "family": "regulatory_network_control"
    },
    {
        "source_text": "Immune system memory cells retain information about past pathogens for rapid response upon re-exposure.",
        "source_domain": 1,
        "target_text": "Cache memory retains frequently accessed data for rapid retrieval without accessing slower storage.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 18,
        "family": "memory_cache_rapid_response"
    },
    {
        "source_text": "Protein folding determines three-dimensional structure from linear amino acid sequence through energy minimization.",
        "source_domain": 1,
        "target_text": "Neural architecture search determines optimal network topology from computational graph through performance optimization.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 19,
        "family": "structure_search_optimization"
    },
    {
        "source_text": "Lateral inhibition in neural circuits creates contrast enhancement by suppressing activation in neighboring neurons.",
        "source_domain": 1,
        "target_text": "Non-maximum suppression in object detection removes overlapping detections by keeping only local maximum confidence scores.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 20,
        "family": "lateral_inhibition_nms"
    },
    # ====== ХИМИЯ ↔ АСТРОФИЗИКА ======
    {
        "source_text": "Catalysis lowers activation energy barrier enabling chemical reactions that would not proceed spontaneously.",
        "source_domain": 3,
        "target_text": "Gravitational lensing bends light paths enabling observation of distant objects that would not otherwise be visible.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 21,
        "family": "pathway_enabling_mechanism"
    },
    {
        "source_text": "Stellar nucleosynthesis fuses light elements into heavier ones inside stars, releasing energy through mass-energy conversion.",
        "source_domain": 0,
        "target_text": "Nuclear fusion reactors fuse light isotopes into heavier ones in plasma, releasing energy through mass-energy conversion.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 22,
        "family": "fusion_nucleosynthesis"
    },
    {
        "source_text": "Polymerization chains monomers into long macromolecular structures through sequential covalent bond formation.",
        "source_domain": 3,
        "target_text": "Galaxy filament formation chains galaxies into cosmic web structures through sequential gravitational accretion.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 23,
        "family": "chain_growth_accretion"
    },
    {
        "source_text": "Crystallization drives molecules from disordered solution into highly ordered periodic solid structure.",
        "source_domain": 3,
        "target_text": "Star formation drives gas from diffuse interstellar medium into dense ordered protostellar structure.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 24,
        "family": "condensation_ordering"
    },
    {
        "source_text": "Combustion rapidly oxidizes fuel releasing stored chemical energy as heat and light.",
        "source_domain": 3,
        "target_text": "Supernova rapidly fuses heavy elements releasing stored gravitational energy as radiation and kinetic energy.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 25,
        "family": "rapid_energy_release"
    },
    # ====== FLUID DYNAMICS ↔ NETWORK THEORY ======
    {
        "source_text": "Turbulent flow exhibits chaotic velocity fluctuations with energy cascading from large to small eddies.",
        "source_domain": 0,
        "target_text": "Internet congestion exhibits chaotic packet delays with traffic cascading through router buffers.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 26,
        "family": "turbulence_congestion"
    },
    {
        "source_text": "Laminar flow transitions to turbulence above critical Reynolds number through instability amplification.",
        "source_domain": 0,
        "target_text": "Orderly network traffic transitions to congestion above critical utilization through bottleneck formation.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 27,
        "family": "laminar_orderly_transition"
    },
    {
        "source_text": "Hydraulic resistance in pipes follows Hagen-Poiseuille law with pressure drop proportional to flow rate.",
        "source_domain": 0,
        "target_text": "Electrical resistance in conductors follows Ohm's law with voltage drop proportional to current.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 28,
        "family": "resistance_ohm_poiseuille"
    },
    # ====== CONTROL THEORY ↔ BIOLOGY ======
    {
        "source_text": "PID controller adjusts output by summing proportional, integral, and derivative responses to error signal.",
        "source_domain": 0,
        "target_text": "Autonomic nervous system adjusts physiology by integrating fast sympathetic, slow parasympathetic responses to homeostatic error.",
        "target_domain": 1,
        "relation": "positive",
        "group_id": max_group + 29,
        "family": "pid_autonomic_control"
    },
    # ====== INFORMATION THEORY ↔ THERMODYNAMICS ======
    {
        "source_text": "Shannon entropy quantifies information uncertainty as sum of negative log probabilities over all possible messages.",
        "source_domain": 2,
        "target_text": "Boltzmann entropy quantifies thermodynamic disorder as sum of negative log probabilities over all microstates.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 30,
        "family": "shannon_boltzmann_entropy"
    },
    {
        "source_text": "Lossless compression removes redundancy from data to produce minimal representation preserving all information.",
        "source_domain": 2,
        "target_text": "Free energy minimization removes thermodynamic redundancy to produce minimum energy configuration.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 31,
        "family": "compression_free_energy"
    },
    {
        "source_text": "Channel capacity limits information transmission rate over noisy channel regardless of encoding strategy.",
        "source_domain": 2,
        "target_text": "Carnot efficiency limits work extraction from heat engine regardless of operating strategy.",
        "target_domain": 0,
        "relation": "positive",
        "group_id": max_group + 32,
        "family": "capacity_efficiency_limit"
    },
    {
        "source_text": "Fourier optics decomposes light field into spatial frequency components that propagate independently.",
        "source_domain": 0,
        "target_text": "Fourier analysis decomposes time signal into frequency components that can be processed independently.",
        "target_domain": 2,
        "relation": "positive",
        "group_id": max_group + 33,
        "family": "fourier_decomposition"
    },
    # ====== HARD NEGATIVES ======
    {
        "source_text": "Quantum tunneling allows particles to pass through classically forbidden energy barriers.",
        "source_domain": 0,
        "target_text": "Database indexing allows queries to skip irrelevant data entries through B-tree structure.",
        "target_domain": 2,
        "relation": "negative",
        "group_id": max_group + 34,
        "family": "neg_quantum_database"
    },
    {
        "source_text": "Backpropagation computes gradients by propagating error signals backwards through layers.",
        "source_domain": 2,
        "target_text": "Crystallization drives molecules from disordered solution into ordered periodic structure.",
        "target_domain": 3,
        "relation": "negative",
        "group_id": max_group + 35,
        "family": "neg_backprop_crystal"
    },
    {
        "source_text": "Shannon entropy quantifies information uncertainty over all possible messages.",
        "source_domain": 2,
        "target_text": "Lateral inhibition creates contrast enhancement by suppressing neighboring neurons.",
        "target_domain": 1,
        "relation": "negative",
        "group_id": max_group + 36,
        "family": "neg_entropy_lateral"
    },
    {
        "source_text": "Market bubbles form when positive feedback loops drive asset prices above fundamental values.",
        "source_domain": 2,
        "target_text": "DNA replication uses template strand to create identical copy with proofreading.",
        "target_domain": 1,
        "relation": "negative",
        "group_id": max_group + 37,
        "family": "neg_bubble_dna"
    },
    {
        "source_text": "Combustion rapidly oxidizes fuel releasing stored chemical energy as heat and light.",
        "source_domain": 3,
        "target_text": "Attention mechanism computes weighted importance of all input positions for each output.",
        "target_domain": 2,
        "relation": "negative",
        "group_id": max_group + 38,
        "family": "neg_combustion_attention"
    }
]

# Merge v5 + new pairs
all_pairs = v5 + new_pairs
print(f'Total pairs: {len(all_pairs)}')
pos = sum(1 for x in all_pairs if x['relation'] == 'positive')
neg = sum(1 for x in all_pairs if x['relation'] == 'negative')
print(f'Positive: {pos}, Negative: {neg}')
fams = set(x['family'] for x in all_pairs)
print(f'Families: {len(fams)}')

out = Path('data/real_pairs_v6.json')
out.write_text(json.dumps(all_pairs, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Saved to {out}')
