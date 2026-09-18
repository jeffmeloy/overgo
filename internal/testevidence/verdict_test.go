package testevidence

import (
	"strings"
	"testing"

	"overgo/internal/runrecord"
)

func verdictEvidence(action string) string {
	lines := []string{
		`{"Action":"run","Package":"overgo/internal/example","Test":"TestStable"}`,
		`{"Action":"pass","Package":"overgo/internal/example","Test":"TestStable"}`,
		`{"Action":"run","Package":"overgo/internal/example","Test":"TestWatched"}`,
		`{"Action":"` + action + `","Package":"overgo/internal/example","Test":"TestWatched"}`,
		`{"Action":"pass","Package":"overgo/internal/example"}`,
	}
	return strings.Join(lines, "\n") + "\n"
}

// TestVerdictClassificationAndRepeatAgreement pins command classification and repeat identity.
func TestVerdictClassificationAndRepeatAgreement(t *testing.T) {
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

	if err := RepeatAgreement(verdictEvidence("pass"), verdictEvidence("pass")); err != nil {
		t.Fatalf("identical runs disagreed: %v", err)
	}
	err := RepeatAgreement(verdictEvidence("pass"), verdictEvidence("fail"))
	if err == nil || !strings.Contains(err.Error(), "TestWatched") ||
		!strings.Contains(err.Error(), "pass") || !strings.Contains(err.Error(), "fail") {
		t.Fatalf("flipped verdict not named: %v", err)
	}
	missing := strings.ReplaceAll(verdictEvidence("pass"), `"Test":"TestWatched"`, `"Test":"TestStable"`)
	if err := RepeatAgreement(verdictEvidence("pass"), missing); err == nil ||
		!strings.Contains(err.Error(), "absent") {
		t.Fatalf("vanished test not reported as absent: %v", err)
	}
	if err := RepeatAgreement("not json", verdictEvidence("pass")); err == nil {
		t.Fatal("undecodable first evidence accepted")
	}
}

func TestRepeatAgreementForMixedVerifier(t *testing.T) {
	command := "go test ./internal/repoanalysis -run '^TestExample$' && go run ./cmd/test-lane ./..."
	first := verdictEvidence("pass") + "test-lane: PASS packages=1 tests=1\n"
	second := verdictEvidence("pass") + "test-lane: PASS packages=1 tests=1\n"
	if err := RepeatAgreementForCommand(command, first, second); err != nil {
		t.Fatal(err)
	}
	if err := RepeatAgreementForCommand(command, first+"{malformed\n", second); err == nil {
		t.Fatal("malformed JSON-looking auxiliary output was accepted")
	}
}
