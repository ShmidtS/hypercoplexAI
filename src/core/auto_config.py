"""
HDIM AutoConfig — Automatic parameter derivation and validation.

Design Principle:
    Explicit parameters ALWAYS override computed ones.
    Computed values derive from minimal input set.

Derivation Chain:
    encoder_type → encoder_output_dim → hidden_dim
    clifford_p,q,r → clifford_dim
    clifford_dim → expert_hidden_dim, memory_key_dim
    expert_names → num_experts

Usage:
    >>> from src.core.auto_config import AutoConfig
    >>> cfg = AutoConfig(encoder_type="modernbert", expert_names=["math", "code"])
    >>> cfg.hidden_dim  # 768 (from ModernBERT)
    >>> cfg.clifford_dim  # 32 (2^5 for Cl(4,1,0))
    >>> cfg.num_experts  # 2 (len(expert_names))
    
    >>> # Convert to HDIMConfig for backward compatibility
    >>> hdim_cfg = cfg.to_hdim_config()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.hdim_model import HDIMConfig
    from src.core.moe_kernel import MoEKernelConfig
    from src.training.experiment_config import ExperimentConfig


# ============================================================
# Encoder dimension constants
# ============================================================

ENCODER_DIMS: Dict[str, int] = {
    # SBERT models
    "sbert": 768,
    "paraphrase-multilingual-mpnet-base-v2": 768,
    "all-minilm-l6-v2": 384,
    "all-mpnet-base-v2": 768,
    
    # ModernBERT models
    "modernbert": 768,
    "modernbert-base": 768,
    "modernbert-large": 1024,
    "answerdotai/modernbert-base": 768,
    "answerdotai/modernbert-large": 1024,
    
    # BERT variants
    "bert-base": 768,
    "bert-large": 1024,
    "roberta-base": 768,
    "roberta-large": 1024,
}

# Default dimension per encoder type
DEFAULT_ENCODER_DIMS: Dict[str, int] = {
    "sbert": 768,
    "modernbert": 768,
    "custom": 128,  # sensible default for custom encoders
}


# ============================================================
# Compute functions (pure, testable)
# ============================================================

def compute_clifford_dim(p: int, q: int, r: int) -> int:
    """
    Compute Clifford algebra dimension.
    
    The dimension of Cl_{p,q,r}(R) is 2^n where n = p + q + r.
    
    Args:
        p: Number of positive basis vectors (e_i^2 = +1)
        q: Number of negative basis vectors (e_i^2 = -1)
        r: Number of nilpotent basis vectors (e_i^2 = 0)
    
    Returns:
        Dimension of the algebra (2^(p+q+r))
    
    Raises:
        ValueError: If signature is invalid (any component negative)
    
    Example:
        >>> compute_clifford_dim(3, 1, 0)  # Cl(3,1,0) = spacetime algebra
        16
        >>> compute_clifford_dim(4, 1, 0)  # Phase 25 signature
        32
    """
    n = p + q + r
    if p < 0 or q < 0 or r < 0:
        raise ValueError(f"Invalid signature: p={p}, q={q}, r={r} (all must be >= 0)")
    return 2 ** n


def compute_expert_hidden_dim(clifford_dim: int, multiplier: int = 2) -> int:
    """
    Compute default expert hidden dimension.
    
    Convention: expert_hidden_dim = clifford_dim * multiplier
    
    Rationale: Standard bottleneck FFN uses expansion factor 2-4.
    For HDIM, clifford_dim is the natural "input" dimension for experts,
    so a 2x expansion provides sufficient representational capacity.
    
    Args:
        clifford_dim: Dimension of Clifford algebra
        multiplier: Expansion factor (default 2)
    
    Returns:
        Expert hidden dimension
    
    Example:
        >>> compute_expert_hidden_dim(32)  # Cl(4,1,0)
        64
    """
    if clifford_dim <= 0:
        raise ValueError(f"clifford_dim must be positive, got {clifford_dim}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    return clifford_dim * multiplier


def compute_memory_key_dim(clifford_dim: int, divisor: int = 2) -> int:
    """
    Compute default memory key dimension.
    
    Convention: memory_key_dim = clifford_dim // divisor
    
    Rationale: Memory stores compressed representations. A smaller key
    dimension reduces memory footprint while preserving enough information
    for retrieval.
    
    Args:
        clifford_dim: Dimension of Clifford algebra
        divisor: Compression factor (default 2)
    
    Returns:
        Memory key dimension (minimum 1)
    
    Example:
        >>> compute_memory_key_dim(32)
        16
    """
    if clifford_dim <= 0:
        raise ValueError(f"clifford_dim must be positive, got {clifford_dim}")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return max(1, clifford_dim // divisor)


def compute_num_experts(
    expert_names: Optional[List[str]],
    explicit_num: Optional[int],
) -> int:
    """
    Derive num_experts from expert_names or use explicit value.
    
    Priority:
    1. If expert_names is provided: len(expert_names)
    2. If explicit_num is provided: explicit_num
    3. Default: 4
    
    Args:
        expert_names: List of expert domain names
        explicit_num: Explicit number of experts
    
    Returns:
        Number of experts
    
    Raises:
        ValueError: If both provided and conflicting
    
    Example:
        >>> compute_num_experts(["math", "code"], None)
        2
        >>> compute_num_experts(None, 8)
        8
        >>> compute_num_experts(None, None)
        4
    """
    if expert_names is not None:
        computed = len(expert_names)
        if explicit_num is not None and explicit_num != computed:
            raise ValueError(
                f"num_experts={explicit_num} conflicts with "
                f"len(expert_names)={computed}. "
                f"Either remove num_experts or ensure it matches len(expert_names)."
            )
        return computed
    if explicit_num is not None:
        return explicit_num
    return 4  # sensible default


def get_encoder_dim(
    encoder_type: str,
    encoder_name: Optional[str] = None,
) -> int:
    """
    Get output dimension for a given encoder.
    
    Args:
        encoder_type: "sbert", "modernbert", or "custom"
        encoder_name: Specific model name (overrides encoder_type default)
    
    Returns:
        Encoder output dimension
    
    Raises:
        ValueError: If encoder_name is provided but unknown
    
    Example:
        >>> get_encoder_dim("modernbert")
        768
        >>> get_encoder_dim("sbert", "all-MiniLM-L6-v2")
        384
    """
    if encoder_name is not None:
        # Try exact match first (case-insensitive)
        encoder_name_lower = encoder_name.lower()
        for key, dim in ENCODER_DIMS.items():
            if key.lower() == encoder_name_lower:
                return dim
        # Try substring match
        for key, dim in ENCODER_DIMS.items():
            if key.lower() in encoder_name_lower or encoder_name_lower in key.lower():
                return dim
        # Unknown encoder - caller must provide hidden_dim explicitly
        raise ValueError(
            f"Unknown encoder_name '{encoder_name}'. "
            f"Known encoders: {sorted(ENCODER_DIMS.keys())}. "
            f"For custom encoders, set encoder_type='custom' and specify hidden_dim explicitly."
        )
    
    return DEFAULT_ENCODER_DIMS.get(encoder_type, 768)


def validate_quaternion_dim(dim: int, context: str = "hidden_dim") -> List[str]:
    """
    Validate dimension for quaternion operations.
    
    QuaternionLinear requires dimension % 4 == 0.
    
    Args:
        dim: Dimension to validate
        context: Parameter name for error message
    
    Returns:
        List of error messages (empty if valid)
    
    Example:
        >>> validate_quaternion_dim(768)
        []
        >>> validate_quaternion_dim(100)
        ['hidden_dim=100 must be divisible by 4...']
    """
    errors = []
    if dim is not None and dim % 4 != 0:
        lower = dim - (dim % 4)
        upper = dim + (4 - dim % 4)
        errors.append(
            f"{context}={dim} must be divisible by 4 for quaternion operations "
            "(QuaternionLinear). "
            f"Nearest valid values: {lower} or {upper}"
        )
    return errors


# ============================================================
# AutoConfig dataclass
# ============================================================

@dataclass
class AutoConfig:
    """
    Auto-deriving configuration for HDIM models.
    
    All derived fields are computed in __post_init__ and can be
    accessed after initialization. Explicit values always override
    computed defaults.
    
    Derivation Rules:
    1. encoder_output_dim ← encoder_type/name lookup (or hidden_dim if unknown)
    2. hidden_dim ← encoder_output_dim (if None)
    3. clifford_dim ← 2^(p+q+r)
    4. num_experts ← len(expert_names) or explicit or 4
    5. expert_hidden_dim ← clifford_dim * 2 (if None)
    6. memory_key_dim ← clifford_dim // 2 (if None)
    
    Validation Rules:
    1. hidden_dim % 4 == 0 (quaternion compatibility)
    2. clifford_dim >= 16 (minimum for meaningful representations)
    3. hidden_dim <= clifford_dim * 16 (avoid projection bottlenecks)
    
    Attributes:
        encoder_type: Type of text encoder (sbert, modernbert, custom)
        encoder_name: Specific model name (overrides encoder_type default)
        hidden_dim: Model hidden dimension (None → encoder_output_dim)
        clifford_p: Positive basis vectors for Cl_{p,q,r}
        clifford_q: Negative basis vectors
        clifford_r: Nilpotent basis vectors
        expert_names: List of expert domain names
        num_experts: Number of experts (None → len(expert_names) or 4)
        expert_hidden_dim: Expert FFN hidden dimension (None → clifford_dim * 2)
        memory_key_dim: Memory key dimension (None → clifford_dim // 2)
        strict_validation: If True, enforce all validation rules (default True)
    
    Example:
        >>> cfg = AutoConfig(encoder_type="modernbert", expert_names=["math", "code"])
        >>> cfg.hidden_dim
        768
        >>> cfg.clifford_dim
        32
        >>> cfg.num_experts_resolved
        2
        >>> hdim_cfg = cfg.to_hdim_config()
    """
    
    # === Primary inputs ===
    encoder_type: Literal["sbert", "modernbert", "custom"] = "sbert"
    encoder_name: Optional[str] = None
    
    # === Overrideable dimensions ===
    hidden_dim: Optional[int] = None
    clifford_p: int = 4
    clifford_q: int = 1
    clifford_r: int = 0
    
    # === MoE parameters ===
    expert_names: Optional[List[str]] = None
    num_experts: Optional[int] = None
    expert_hidden_dim: Optional[int] = None
    memory_key_dim: Optional[int] = None
    
    # === Validation control ===
    strict_validation: bool = True
    
    # === Derived fields (computed in __post_init__) ===
    encoder_output_dim: int = field(init=False, repr=False)
    clifford_dim: int = field(init=False, repr=False)
    _hidden_dim_resolved: int = field(init=False, repr=False)
    _num_experts: int = field(init=False, repr=False)
    _expert_hidden_dim: int = field(init=False, repr=False)
    _memory_key_dim: int = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Compute all derived fields and validate constraints."""
        self._compute_encoder_output_dim()
        self._compute_hidden_dim()
        self._compute_clifford_dim()
        self._compute_num_experts()
        self._compute_expert_dims()
        if self.strict_validation:
            self._validate_constraints()
    
    # ----------------------------------------------------------
    # Derivation methods (private)
    # ----------------------------------------------------------
    
    def _compute_encoder_output_dim(self) -> None:
        """Derive encoder output dimension from encoder_type/name."""
        # Try to get encoder dim from name/type
        # If unknown encoder_name and hidden_dim provided, use hidden_dim
        try:
            self.encoder_output_dim = get_encoder_dim(self.encoder_type, self.encoder_name)
        except ValueError:
            # Unknown encoder - use hidden_dim if provided, else re-raise
            if self.hidden_dim is not None:
                self.encoder_output_dim = self.hidden_dim
            else:
                raise
    
    def _compute_hidden_dim(self) -> None:
        """Derive hidden_dim from encoder or use explicit value."""
        self._hidden_dim_resolved = self.hidden_dim if self.hidden_dim is not None else self.encoder_output_dim
    
    def _compute_clifford_dim(self) -> None:
        """Compute Clifford algebra dimension."""
        self.clifford_dim = compute_clifford_dim(
            self.clifford_p, self.clifford_q, self.clifford_r
        )
    
    def _compute_num_experts(self) -> None:
        """Derive num_experts from expert_names or use explicit value."""
        self._num_experts = compute_num_experts(self.expert_names, self.num_experts)
    
    def _compute_expert_dims(self) -> None:
        """Derive expert_hidden_dim and memory_key_dim from clifford_dim."""
        self._expert_hidden_dim = (
            self.expert_hidden_dim
            if self.expert_hidden_dim is not None
            else compute_expert_hidden_dim(self.clifford_dim)
        )
        self._memory_key_dim = (
            self.memory_key_dim
            if self.memory_key_dim is not None
            else compute_memory_key_dim(self.clifford_dim)
        )
    
    # ----------------------------------------------------------
    # Validation (private)
    # ----------------------------------------------------------
    
    def _validate_constraints(self) -> None:
        """Cross-parameter validation with helpful error messages."""
        errors: List[str] = []
        
        # Quaternion compatibility
        errors.extend(validate_quaternion_dim(self._hidden_dim_resolved, "hidden_dim"))
        
        # Clifford dimension reasonableness
        if self.clifford_dim < 16:
            errors.append(
                f"clifford_dim={self.clifford_dim} is very small. "
                "Minimum recommended: 16 (Cl(3,1,0) or Cl(4,0,0)). "
                f"Current signature: Cl({self.clifford_p},{self.clifford_q},{self.clifford_r})"
            )
        
        # Hidden dim vs clifford_dim alignment (relaxed: warn only if hidden >> clifford)
        # For encoders like ModernBERT (768), clifford_dim can be smaller - this is OK
        # Only warn if hidden_dim > clifford_dim * 16 (very aggressive projection)
        if self._hidden_dim_resolved > self.clifford_dim * 16:
            recommended_n = max(4, int(math.log2(self._hidden_dim_resolved)))
            errors.append(
                f"hidden_dim={self._hidden_dim_resolved} >> clifford_dim={self.clifford_dim}: "
                "aggressive projection may lose information. "
                f"Consider p+q+r >= {recommended_n} or set strict_validation=False."
            )
        
        # Memory key dimension bounds
        if self._memory_key_dim < 8 and self.clifford_dim >= 16:
            errors.append(
                f"memory_key_dim={self._memory_key_dim} is very small. "
                "This may limit memory retrieval quality. "
                "Consider increasing clifford_dim or setting memory_key_dim explicitly."
            )
        
        if errors:
            raise ValueError(
                "AutoConfig validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    # ----------------------------------------------------------
    # Properties for derived values
    # ----------------------------------------------------------
    
    @property
    def hidden_dim_resolved(self) -> int:
        """Resolved hidden dimension (computed or explicit)."""
        return self._hidden_dim_resolved
    
    @property
    def num_experts_resolved(self) -> int:
        """Resolved number of experts (computed or explicit)."""
        return self._num_experts
    
    @property
    def expert_hidden_dim_resolved(self) -> int:
        """Resolved expert hidden dimension (computed or explicit)."""
        return self._expert_hidden_dim
    
    @property
    def memory_key_dim_resolved(self) -> int:
        """Resolved memory key dimension (computed or explicit)."""
        return self._memory_key_dim
    
    # ----------------------------------------------------------
    # Conversion methods
    # ----------------------------------------------------------
    
    def to_hdim_config(self, **overrides: Any) -> "HDIMConfig":
        """
        Convert AutoConfig to HDIMConfig for backward compatibility.
        
        Args:
            **overrides: Additional HDIMConfig fields to override
                (dropout, top_k, memory_type, etc.)
                Note: num_domains should be passed here if needed.
        
        Returns:
            HDIMConfig instance with derived parameters
        
        Example:
            >>> cfg = AutoConfig(encoder_type="modernbert")
            >>> hdim_cfg = cfg.to_hdim_config(num_domains=8, dropout=0.2)
        """
        from src.models.hdim_model import HDIMConfig
        
        # Reserved fields that are set from AutoConfig
        reserved = {
            "hidden_dim", "num_experts", "expert_names",
            "clifford_p", "clifford_q", "clifford_r",
            "memory_key_dim",
        }
        
        # Extract num_domains separately to avoid double-passing
        num_domains = overrides.pop("num_domains", 4)
        
        return HDIMConfig(
            hidden_dim=self._hidden_dim_resolved,
            num_domains=num_domains,
            num_experts=self._num_experts,
            expert_names=self.expert_names,
            clifford_p=self.clifford_p,
            clifford_q=self.clifford_q,
            clifford_r=self.clifford_r,
            memory_key_dim=self._memory_key_dim,
            **{k: v for k, v in overrides.items() if k not in reserved},
        )
    
    def to_moe_kernel_config(self, **overrides: Any) -> "MoEKernelConfig":
        """
        Convert AutoConfig to MoEKernelConfig.
        
        Args:
            **overrides: Additional MoEKernelConfig fields to override
                (temperature, z_loss_weight, use_shared_expert, etc.)
        
        Returns:
            MoEKernelConfig instance with derived parameters
        
        Example:
            >>> cfg = AutoConfig(expert_names=["math", "code", "science"])
            >>> moe_cfg = cfg.to_moe_kernel_config(use_shared_expert=False)
        """
        from src.core.moe_kernel import MoEKernelConfig
        
        return MoEKernelConfig(
            input_dim=self.clifford_dim,
            expert_hidden_dim=self._expert_hidden_dim,
            num_experts=self._num_experts,
            expert_names=self.expert_names,
            **overrides,
        )
    
    def to_experiment_config(self, **overrides: Any) -> "ExperimentConfig":
        """
        Convert AutoConfig to ExperimentConfig for training.
        
        Args:
            **overrides: Additional ExperimentConfig fields
                (epochs, batch_size, lr, etc.)
        
        Returns:
            ExperimentConfig instance
        
        Example:
            >>> cfg = AutoConfig(encoder_type="modernbert")
            >>> exp_cfg = cfg.to_experiment_config(epochs=10, batch_size=32)
        """
        from src.training.experiment_config import ExperimentConfig
        
        return ExperimentConfig(
            hidden_dim=self._hidden_dim_resolved,
            num_experts=self._num_experts,
            expert_names=self.expert_names,
            **overrides,
        )
    
    # ----------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------
    
    @classmethod
    def from_encoder(
        cls,
        encoder_type: str = "sbert",
        encoder_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "AutoConfig":
        """
        Create AutoConfig from encoder specification.
        
        Args:
            encoder_type: "sbert", "modernbert", or "custom"
            encoder_name: Specific model name
            **kwargs: Additional AutoConfig fields
        
        Returns:
            AutoConfig with auto-derived hidden_dim
        
        Example:
            >>> cfg = AutoConfig.from_encoder("modernbert", expert_names=["math"])
        """
        return cls(encoder_type=encoder_type, encoder_name=encoder_name, **kwargs)
    
    @classmethod
    def from_clifford_signature(
        cls,
        p: int,
        q: int,
        r: int = 0,
        hidden_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> "AutoConfig":
        """
        Create AutoConfig from Clifford algebra signature.
        
        Args:
            p: Positive basis vectors
            q: Negative basis vectors  
            r: Nilpotent basis vectors
            hidden_dim: Override hidden_dim (None → clifford_dim)
            **kwargs: Additional AutoConfig fields
        
        Returns:
            AutoConfig with specified Clifford signature
        
        Example:
            >>> cfg = AutoConfig.from_clifford_signature(3, 1, 0)  # Cl(3,1,0)
            >>> cfg.clifford_dim
            16
        """
        clifford_dim = compute_clifford_dim(p, q, r)
        return cls(
            clifford_p=p,
            clifford_q=q,
            clifford_r=r,
            hidden_dim=hidden_dim if hidden_dim is not None else clifford_dim,
            **kwargs,
        )
    
    @classmethod
    def from_hdim_config(cls, hdim_cfg: "HDIMConfig") -> "AutoConfig":
        """
        Create AutoConfig from existing HDIMConfig.
        
        This is useful for upgrading existing configurations to use
        auto-derivation features.
        
        Args:
            hdim_cfg: Existing HDIMConfig instance
        
        Returns:
            AutoConfig with values from HDIMConfig
        
        Example:
            >>> from src.models.hdim_model import HDIMConfig
            >>> old_cfg = HDIMConfig(hidden_dim=128, num_experts=8)
            >>> auto_cfg = AutoConfig.from_hdim_config(old_cfg)
        """
        return cls(
            hidden_dim=hdim_cfg.hidden_dim,
            clifford_p=hdim_cfg.clifford_p,
            clifford_q=hdim_cfg.clifford_q,
            clifford_r=hdim_cfg.clifford_r,
            expert_names=hdim_cfg.expert_names,
            num_experts=hdim_cfg.num_experts,
            memory_key_dim=hdim_cfg.memory_key_dim,
        )
    
    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------
    
    def __repr__(self) -> str:
        """Compact representation showing key resolved values."""
        return (
            f"AutoConfig("
            f"hidden_dim={self._hidden_dim_resolved}, "
            f"clifford_dim={self.clifford_dim} [Cl({self.clifford_p},{self.clifford_q},{self.clifford_r})], "
            f"num_experts={self._num_experts}, "
            f"expert_hidden_dim={self._expert_hidden_dim}, "
            f"memory_key_dim={self._memory_key_dim}"
            f")"
        )
    
    def summary(self) -> str:
        """
        Return detailed summary of configuration.
        
        Returns:
            Multi-line string with all parameters
        
        Example:
            >>> print(cfg.summary())
        """
        lines = [
            "AutoConfig Summary:",
            f"  Encoder: {self.encoder_type}" + (f" ({self.encoder_name})" if self.encoder_name else ""),
            f"  encoder_output_dim: {self.encoder_output_dim}",
            f"  hidden_dim: {self._hidden_dim_resolved}",
            f"  Clifford: Cl({self.clifford_p},{self.clifford_q},{self.clifford_r}) → dim={self.clifford_dim}",
            f"  MoE:",
            f"    num_experts: {self._num_experts}",
            f"    expert_names: {self.expert_names or 'auto-generated'}",
            f"    expert_hidden_dim: {self._expert_hidden_dim}",
            f"  Memory:",
            f"    memory_key_dim: {self._memory_key_dim}",
        ]
        return "\n".join(lines)


# ============================================================
# Module exports
# ============================================================

__all__ = [
    # Constants
    "ENCODER_DIMS",
    "DEFAULT_ENCODER_DIMS",
    # Compute functions
    "compute_clifford_dim",
    "compute_expert_hidden_dim",
    "compute_memory_key_dim",
    "compute_num_experts",
    "get_encoder_dim",
    "validate_quaternion_dim",
    # Main class
    "AutoConfig",
]
