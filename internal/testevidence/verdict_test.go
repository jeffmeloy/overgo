package testevidence

import (
	"testing"

	"overgo/internal/runrecord"
)

// TestVerdictClassification pins command classification.
func TestVerdictClassification(t *testing.T) {
	classifications := []struct {
		command string
		want    runrecord.VerdictClass
	}{
		{"go test ./internal/repodb -run '^TestStreamingSnapshotRoundTrip$' -count=1 -v", runrecord.VerdictBitwiseDeterministic},
		{"grep -q 'modeltest' .github/workflows/test.yml", runrecord.VerdictBitwiseDeterministic},
		{"OVERGO_CUDA_TEST=1 go test ./internal/adaptiveparity -run '^TestComponentCompositionViability$' -count=1 -v", runrecord.VerdictToleranceBounded},
		{"OVERGO_SEALED_AUTHORITY_TEST=1 go test ./internal/protection -count=1", runrecord.VerdictToleranceBounded},
		{"go run ./cmd/device-lane", runrecord.VerdictToleranceBounded},
		{"OVERGO_SEEDS=3 go test ./internal/scratchmodel -run '^TestMultiSeedGain$' -count=1", runrecord.VerdictStochasticMultiSeed},
		{"go test ./internal/densecausal -run '^TestMultiSeedLossEnvelope$' -count=1 -v", runrecord.VerdictStochasticMultiSeed},
	}
	for _, c := range classifications {
		if got := runrecord.ClassifyVerifyCommand(c.command); got != c.want {
			t.Errorf("ClassifyVerifyCommand(%q) = %s, want %s", c.command, got, c.want)
		}
		got, err := runrecord.ClassifyVerifyCommandForPolicy(runrecord.VerifyPolicyV1, c.command)
		if err != nil || got != c.want {
			t.Errorf("ClassifyVerifyCommandForPolicy(v1, %q) = (%s, %v), want %s", c.command, got, err, c.want)
		}
	}
	if _, err := runrecord.ClassifyVerifyCommandForPolicy(runrecord.VerifyPolicy("verify-classifier-v2"), "go test ./..."); err == nil {
		t.Fatal("unknown future verify classifier policy accepted")
	}
}
