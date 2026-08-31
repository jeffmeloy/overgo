package gate

import "testing"

func TestGateAlwaysRunsModernGoRatchet(t *testing.T) {
	checks := (&gateContext{}).pipelineChecks()
	for _, check := range checks {
		if check.Descriptor.Name == "modern-go" {
			if !check.Descriptor.Always {
				t.Fatal("modern-Go ratchet is not an always-run gate check")
			}
			return
		}
	}
	t.Fatal("modern-Go ratchet is absent from the gate pipeline")
}
