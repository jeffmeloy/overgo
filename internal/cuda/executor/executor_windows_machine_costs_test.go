//go:build windows

package executor

import (
	"math/rand/v2"
	"testing"
	"time"

	"overgo/internal/hostexpert"
	"overgo/internal/tensor/dtype"
)

// The measurement's fixture: Flash-Next's expert widths and formats (IQ3_S gate and up, IQ4_NL down), and the bounds a
// measurement on this card's link and CPU must fall in, and its cost at load.
const (
	machineFixtureHidden       = 2560
	machineFixtureIntermediate = 640
	machineLinkSlowest         = 1e9
	machineLinkFastest         = 1e11
	machineMeasureMost         = time.Second
)

// TestLoadMeasuresTheMachineCosts measures this machine's costs as a model's load does, on an expert of Flash-Next's
// widths and formats in host memory: the link's rate copying it to the card, the CPU's fixed time a call and a slot's.
// Each is a positive rate of this hardware's order, the measurement takes a fraction of a load, and the costs place the
// expert's crossover between the CPU and a copy within the slots the staging reserves (the constant it replaces held 32;
// the comments' figures were ~160 us a copy and ~9 us a slot).
func TestLoadMeasuresTheMachineCosts(t *testing.T) {
	e := newFixtureExecutor(t)
	random := rand.New(rand.NewPCG(3, 5))
	bank := func(kind dtype.Type, rows, width uint64) []byte {
		bytes, err := kind.StorageBytes(rows*width, width)
		if err != nil {
			t.Fatal(err)
		}
		data := make([]byte, bytes)
		for index := range data {
			data[index] = byte(random.IntN(1 << 8))
		}
		return data
	}
	expert := hostexpert.Expert{
		Gate: bank(dtype.IQ3S, machineFixtureIntermediate, machineFixtureHidden),
		Up:   bank(dtype.IQ3S, machineFixtureIntermediate, machineFixtureHidden),
		Down: bank(dtype.IQ4NL, machineFixtureHidden, machineFixtureIntermediate),
	}
	formats := hostexpert.Formats{Gate: dtype.IQ3S, Up: dtype.IQ3S, Down: dtype.IQ4NL}
	started := time.Now()
	costs, err := e.MeasureHostExpertCosts(t.Context(), expert, formats, machineFixtureHidden, machineFixtureIntermediate)
	took := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	bytes := uint64(len(expert.Gate) + len(expert.Up) + len(expert.Down))
	limit := costs.hostSlotLimit(bytes)
	t.Logf("link %.2f GB/s (the expert's %d bytes in %v), CPU %v a call and %v a slot: the CPU's up to %d slots; measured in %v",
		costs.LinkBytesPerSecond/machineLinkSlowest, bytes, time.Duration(float64(bytes)/costs.LinkBytesPerSecond*float64(time.Second)),
		costs.CPUFixed, costs.CPUSlot, limit, took)
	if costs.LinkBytesPerSecond < machineLinkSlowest || costs.LinkBytesPerSecond > machineLinkFastest || costs.CPUSlot <= 0 || costs.CPUFixed < 0 {
		t.Fatalf("the measured costs %+v are not this hardware's", costs)
	}
	if took > machineMeasureMost {
		t.Fatalf("the measurement took %v at load, want under %v", took, machineMeasureMost)
	}
	if limit == 0 || limit > hostExpertSlotsMost {
		t.Fatalf("the costs put the expert's crossover at %d slots, want between 1 and %d", limit, hostExpertSlotsMost)
	}
}
