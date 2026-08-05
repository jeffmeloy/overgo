package model

// ValidationPolicy: architecture-owned metadata invariants.
type ValidationPolicy struct {
	AttentionFree     bool
	ClampQKV          bool
	RequireLogitScale bool
	RelativeAttention bool
	RequireDecoder    bool
	MultiAxisRoPE     bool
	BoundDeepstack    bool
}
