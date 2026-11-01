package capabilityruntime

import (
	"errors"
	"testing"
	"time"

	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

type peakMeter struct {
	peak uint64
	err  error
}

func (m peakMeter) PeakDeviceBytes() (uint64, error) { return m.peak, m.err }

// TestPeakIsMeasuredOnlyWhereTheMeterAnswers holds the envelope to what was
// read: a model with no meter and a meter that fails leave the peak absent
// whatever the load took, and a meter's zero is a measurement as its other
// readings are.
func TestPeakIsMeasuredOnlyWhereTheMeterAnswers(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name     string
		model    any
		peak     uint64
		measured bool
	}{
		{name: "no meter", model: struct{}{}},
		{name: "failed read", model: peakMeter{err: errors.New("unread")}},
		{name: "zero", model: peakMeter{}, measured: true},
		{name: "reading", model: peakMeter{peak: 7}, peak: 7, measured: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got := measure("output", test.model, workflowruntime.Result{}, time.Second, recipe.Program{})
			if got.PeakDeviceBytes != test.peak || got.PeakMeasured != test.measured {
				t.Fatalf("peak %d measured %v, want %d and %v", got.PeakDeviceBytes, got.PeakMeasured, test.peak, test.measured)
			}
		})
	}
}
