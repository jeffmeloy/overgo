package densecausal

import (
	"strings"
	"testing"
)

// TestDeviceTrainingSupportedAdmission proves the host-side capability
// predicate refuses a device-unsupported architecture (attention bias) with a
// reason, and admits an otherwise-identical model without bias. This is the
// SELECTION-time gate: it decides device-vs-host from Dims alone, before any
// device session exists, so no GPU is required to run it.
func TestDeviceTrainingSupportedAdmission(t *testing.T) {
	base := Dims{
		Vocab: 8, Hidden: 8, Layers: 1,
		Heads: 2, KVHeads: 1, HeadDim: 4, Intermediate: 16,
		RopeTheta: 10000, RMSEps: 1e-6,
	}

	supported := base
	supported.AttnBias = false
	if ok, reason := DeviceTrainingSupported(supported); !ok {
		t.Fatalf("no-bias model must be admitted to the device path, got refused: %q", reason)
	}

	biased := base
	biased.AttnBias = true
	ok, reason := DeviceTrainingSupported(biased)
	if ok {
		t.Fatal("attention-bias model must be refused by the device training predicate")
	}
	if reason == "" {
		t.Fatal("refusal must carry a non-empty reason")
	}
	if !strings.Contains(strings.ToLower(reason), "bias") {
		t.Fatalf("refusal reason %q should name the unsupported trait (bias)", reason)
	}
}
