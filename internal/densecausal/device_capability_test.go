package densecausal

import "testing"

// TestDeviceTrainingSupportedAdmission proves the host-side capability
// predicate admits both dense causal geometries currently implemented.
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
	if ok, reason := DeviceTrainingSupported(biased); !ok {
		t.Fatalf("attention-bias model must be admitted to the device path, got refused: %q", reason)
	}
}
