#!/usr/bin/env python3
import json

with open('E:/hypercoplexAI/data/real_pairs_v2.json', 'r', encoding='utf-8') as f:
    pairs = json.load(f)

new_pairs = [
  {
    "source_text": "Wing flutter in aircraft: aerodynamic forces couple with structural modes, leading to divergent oscillations above critical speed.",
    "source_domain": 0,
    "target_text": "Parkinsonian tremor: pathological coupling between motor cortex and basal ganglia creates 4-6 Hz oscillations resistant to voluntary control.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 30,
    "family": "pathological_oscillation_coupling"
  },
  {
    "source_text": "Compressed sensing: sparse signals can be exactly recovered from far fewer measurements than Nyquist rate using L1 minimization.",
    "source_domain": 2,
    "target_text": "Sparse coding in visual cortex: natural images represented by small subset of active neurons from overcomplete dictionary minimizing redundancy.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 31,
    "family": "sparse_representation"
  },
  {
    "source_text": "Network centrality measures (PageRank, betweenness): nodes with high centrality disproportionately control information flow in complex networks.",
    "source_domain": 2,
    "target_text": "Hub neurons in connectome: highly connected interneurons coordinate local circuit activity and serve as bottlenecks for signal propagation.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 32,
    "family": "network_hub_control"
  },
  {
    "source_text": "Adaptive optics: wavefront sensor measures and corrects atmospheric distortions in real-time using deformable mirror feedback.",
    "source_domain": 0,
    "target_text": "Eye accommodation: ciliary muscles dynamically adjust lens curvature via feedback from retinal blur detection for sharp focus.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 33,
    "family": "adaptive_feedback_optics"
  },
  {
    "source_text": "Galton-Watson branching process: population survives long-term only when mean offspring number exceeds one, extinction otherwise certain.",
    "source_domain": 2,
    "target_text": "Immune system clonal selection: lymphocyte clones expand only when antigen binding exceeds activation threshold, dying otherwise.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 34,
    "family": "threshold_survival_branching"
  },
  {
    "source_text": "Information bottleneck method: optimal compressed representation preserves maximum information about output while minimizing information about input.",
    "source_domain": 2,
    "target_text": "Sensory adaptation: neural systems discard constant background stimuli and encode only unexpected changes, maximizing informational efficiency.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 35,
    "family": "information_compression_efficiency"
  },
  {
    "source_text": "Lyapunov exponent characterizes rate of exponential divergence of nearby trajectories, positive value indicating chaos.",
    "source_domain": 2,
    "target_text": "Sensitive dependence on initial conditions in weather: small perturbations amplify exponentially, limiting deterministic predictability to weeks.",
    "target_domain": 0,
    "relation": "positive",
    "group_id": 37,
    "family": "chaotic_sensitivity"
  },
  {
    "source_text": "Topological insulators: bulk gap with conducting surface states protected by time-reversal symmetry, robust against non-magnetic disorder.",
    "source_domain": 3,
    "target_text": "Topological defects in embryonic morphogenesis: conserved topological charge of cell orientation fields guides tissue folding and organ placement.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 38,
    "family": "topological_protection"
  },
  {
    "source_text": "Adiabatic theorem: quantum system remains in instantaneous eigenstate when Hamiltonian changes slowly enough relative to energy gap.",
    "source_domain": 3,
    "target_text": "Ecological succession: slow environmental change allows species composition to track quasi-static equilibrium, failing at rapid perturbation.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 39,
    "family": "adiabatic_tracking"
  },
  {
    "source_text": "Piezoelectric effect: mechanical stress in crystalline materials generates electric polarization proportional to applied deformation.",
    "source_domain": 0,
    "target_text": "Mechanotransduction in hair cells: stereocilia deflection opens ion channels converting mechanical force to electrical signal in auditory system.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 42,
    "family": "mechanoelectric_transduction"
  },
  {
    "source_text": "Bifurcation diagram of logistic map: fixed points, period-doubling cascade, and chaos windows as nonlinear parameter varies.",
    "source_domain": 2,
    "target_text": "Developmental bifurcations in cell fate: stem cell commitment follows sequence of symmetry-breaking decisions creating binary fate trees.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 43,
    "family": "bifurcation_fate_tree"
  },
  {
    "source_text": "Pipe network flow distribution: flow partitions at junctions following Kirchhoff current law analogy to minimize pressure drop.",
    "source_domain": 0,
    "target_text": "Murray law in vascular networks: optimal branching ratios minimize energy cost of blood flow, verified in arterial tree geometry.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 44,
    "family": "optimal_network_flow"
  },
  {
    "source_text": "Ising model on a graph: collective magnetization emerges through local spin interactions, critical behavior at phase transition.",
    "source_domain": 3,
    "target_text": "Social consensus formation: local opinion interactions on social network produce collective opinion states with critical tipping points.",
    "target_domain": 2,
    "relation": "positive",
    "group_id": 46,
    "family": "collective_state_formation"
  },
  {
    "source_text": "Soliton propagation in optical fibers: nonlinear self-phase modulation balances group velocity dispersion enabling distortion-free pulse transmission.",
    "source_domain": 3,
    "target_text": "Nerve impulse propagation: Hodgkin-Huxley soliton-like action potential maintains shape through active regeneration at each membrane segment.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 47,
    "family": "soliton_shape_preservation"
  },
  {
    "source_text": "Stochastic resonance: optimal level of noise enhances detection of subthreshold signals in nonlinear systems.",
    "source_domain": 2,
    "target_text": "Neural noise benefits in sensory perception: moderate background neural fluctuations improve detection of weak stimuli near perceptual threshold.",
    "target_domain": 1,
    "relation": "positive",
    "group_id": 49,
    "family": "stochastic_resonance"
  },
  {
    "source_text": "Buckling instability in slender columns occurs when axial load exceeds Euler critical load.",
    "source_domain": 0,
    "target_text": "Bose-Einstein condensation in ultracold atoms condenses particles into single quantum ground state.",
    "target_domain": 3,
    "relation": "negative",
    "group_id": 200,
    "family": "neg_buckling_bec"
  },
  {
    "source_text": "Cavitation erosion in pump impellers damages metal surfaces through pressure transients.",
    "source_domain": 0,
    "target_text": "Bose-Einstein condensation occurs when atoms cool below critical temperature.",
    "target_domain": 3,
    "relation": "negative",
    "group_id": 201,
    "family": "neg_cavitation_bec"
  },
  {
    "source_text": "Topology optimization distributes material for maximum stiffness.",
    "source_domain": 0,
    "target_text": "DNA repair after double-strand breaks uses homologous recombination.",
    "target_domain": 1,
    "relation": "negative",
    "group_id": 202,
    "family": "neg_topo_dna"
  },
  {
    "source_text": "Gradient descent converges for convex functions by following negative gradient direction.",
    "source_domain": 2,
    "target_text": "Superconductivity requires Cooper pairs and phonon-mediated attraction below Tc.",
    "target_domain": 3,
    "relation": "negative",
    "group_id": 203,
    "family": "neg_gd_superconductor"
  },
  {
    "source_text": "Fourier transform decomposes signals into frequency components for spectral analysis.",
    "source_domain": 2,
    "target_text": "Enzyme catalysis lowers activation energy through transition state stabilization.",
    "target_domain": 1,
    "relation": "negative",
    "group_id": 204,
    "family": "neg_fourier_enzyme"
  },
  {
    "source_text": "Turbulent boundary layer transition occurs at critical Reynolds number.",
    "source_domain": 0,
    "target_text": "Random matrix eigenvalue distribution follows Wigner semicircle law.",
    "target_domain": 2,
    "relation": "negative",
    "group_id": 205,
    "family": "neg_turbulence_rmt"
  },
  {
    "source_text": "Heat pipe uses capillary action and phase change for heat transfer.",
    "source_domain": 0,
    "target_text": "Quantum tunneling probability decays exponentially with barrier width.",
    "target_domain": 3,
    "relation": "negative",
    "group_id": 206,
    "family": "neg_heatpipe_tunnel"
  },
  {
    "source_text": "Self-organized criticality in sandpile models produces power-law avalanche distributions.",
    "source_domain": 2,
    "target_text": "Shock waves form in supersonic flow when velocity exceeds sound speed.",
    "target_domain": 0,
    "relation": "negative",
    "group_id": 207,
    "family": "neg_soc_shock"
  },
  {
    "source_text": "Piezoelectric effect generates electric polarization from mechanical stress in crystals.",
    "source_domain": 0,
    "target_text": "Percolation threshold determines emergence of spanning cluster in lattice networks.",
    "target_domain": 2,
    "relation": "negative",
    "group_id": 208,
    "family": "neg_piezo_percolation"
  },
  {
    "source_text": "Compressed sensing recovers sparse signals from few measurements via L1 minimization.",
    "source_domain": 2,
    "target_text": "Cardiac arrhythmia occurs through electrical reentry loops in myocardium.",
    "target_domain": 1,
    "relation": "negative",
    "group_id": 209,
    "family": "neg_compressed_arrhythmia"
  },
  {
    "source_text": "Soliton propagation in optical fibers enables distortion-free pulse transmission over long distance.",
    "source_domain": 3,
    "target_text": "Kalman filter optimally estimates hidden state by combining noisy measurements with model predictions.",
    "target_domain": 2,
    "relation": "negative",
    "group_id": 210,
    "family": "neg_soliton_kalman"
  },
  {
    "source_text": "Topological insulators have bulk gap with surface states protected by time-reversal symmetry.",
    "source_domain": 3,
    "target_text": "Ant colony optimization finds shortest paths through pheromone trail reinforcement.",
    "target_domain": 2,
    "relation": "negative",
    "group_id": 211,
    "family": "neg_topins_aco"
  }
]

pairs.extend(new_pairs)

pos = [p for p in pairs if p['relation'] == 'positive']
neg = [p for p in pairs if p['relation'] == 'negative']
print(f'Total: {len(pairs)}, Positive: {len(pos)}, Negative: {len(neg)}')

with open('E:/hypercoplexAI/data/real_pairs_v2.json', 'w', encoding='utf-8') as f:
    json.dump(pairs, f, indent=2, ensure_ascii=False)
print('Saved successfully.')
