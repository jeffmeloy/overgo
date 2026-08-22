package device

// DefaultOrdinal selects the first CUDA device when no caller-level device
// routing policy is present.
func DefaultOrdinal() int { return 0 }
