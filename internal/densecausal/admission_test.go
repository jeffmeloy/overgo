package densecausal

import "testing"

// DeviceTrainingAdmitted gates the device backward's known constraints so
// backend selection falls back to host instead of failing. Attention-bias
// models (e.g. Qwen2) are loadable but not device-trainable yet.
func TestDeviceTrainingAdmitted(t *testing.T) {
	if ok, reason := (&Model{Dims: Dims{AttnBias: false}}).DeviceTrainingAdmitted(); !ok {
		t.Fatalf("no-bias model should be admitted, got reason %q", reason)
	}
	ok, reason := (&Model{Dims: Dims{AttnBias: true}}).DeviceTrainingAdmitted()
	if ok {
		t.Fatal("attention-bias model must not be admitted to the device path")
	}
	if reason == "" {
		t.Fatal("rejection must include a human-readable reason")
	}
}
