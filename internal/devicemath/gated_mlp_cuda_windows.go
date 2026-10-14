//go:build windows

package devicemath

// GatedMLPGrads holds the gradients of a SwiGLU/gated-MLP block.
type GatedMLPGrads struct {
	DX     []float32 // [rows, d]
	DWGate []float32 // [d, inter]
	DWUp   []float32 // [d, inter]
	DWDown []float32 // [inter, d]
}
