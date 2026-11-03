package main

import (
	"runtime/debug"
	"testing"

	"overgo/internal/hybridtrain"
)

// Qwen3.5-9B's layer-streamed budget, as its proof prints it: its matrix and vector parameters, its largest layer's
// gradient, and the bytes the budget holds of them.
const (
	nineBMatrix  = 6_784_286_720
	nineBScratch = 218_365_952
	nineBVectors = 1_057_280
	nineBBounded = 55_160_444_928
)

// TestTheHybridRouteSetsItsMemoryLimitFromItsBudget: the layer-streamed lane sets the process's Go memory limit to its
// budget, the masters and momentum, one layer's gradient and the vectors, so the route runs with no environment set by
// hand; on Qwen3.5-9B that is 55,160,444,928 bytes, below the 68 GiB its proof was once run under.
//
// Serial: it sets the memory limit every test of the package runs under, and restores it.
func TestTheHybridRouteSetsItsMemoryLimitFromItsBudget(t *testing.T) {
	previous := debug.SetMemoryLimit(-1)
	defer debug.SetMemoryLimit(previous)
	budget := hybridtrain.LayerStreamedBudget{MasterElems: nineBMatrix, MomentumElems: nineBMatrix, ScratchElems: nineBScratch, VectorElems: nineBVectors}
	if limit := setMemoryLimit(budget); limit != nineBBounded || debug.SetMemoryLimit(-1) != nineBBounded {
		t.Fatalf("the route set a memory limit of %d bytes, and the process holds %d, want the budget's %d", limit, debug.SetMemoryLimit(-1), nineBBounded)
	}
}
