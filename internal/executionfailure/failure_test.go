package executionfailure

import "testing"

// TestFailureNormalizationCorpus pins the classifier: a corpus of raw
// evidence maps deterministically onto every canonical cause, the
// classification names the classifier version, and unrecognized
// evidence falls through to process-failure or unknown.
func TestFailureNormalizationContract(t *testing.T) {
	corpus := []struct {
		name     string
		evidence Evidence
		cause    Cause
		rule     string
	}{
		{"context budget", Evidence{Message: "request exceeds the maximum context length of 8192 tokens"}, CauseContextExhaustion, "context-budget"},
		{"stderr context", Evidence{Detail: "server: prompt is too long for the loaded window"}, CauseContextExhaustion, "context-budget"},
		{"configuration", Evidence{Message: "flag provided but not defined: -modl"}, CauseConfiguration, "declared-settings"},
		{"authorization", Evidence{Message: "open store: permission denied"}, CauseAuthorization, "credentials"},
		{"quota", Evidence{Message: "429 too many requests"}, CauseQuota, "allowance"},
		{"capacity", Evidence{Detail: "cuda: out of memory allocating 4096 MiB"}, CauseCapacity, "resources"},
		{"network", Evidence{Message: "dial tcp 127.0.0.1:8080: connection refused", ExitCode: 1}, CauseNetwork, "transport"},
		{"model availability", Evidence{Message: "serve: model not found in catalog"}, CauseModelAvailability, "servable"},
		{"timeout", Evidence{Message: "context deadline exceeded"}, CauseTimeout, "expired-wait"},
		{"missing executable", Evidence{Message: `exec: "nvcc": executable file not found in %PATH%`}, CauseMissingExecutable, "command-lookup"},
		{"process failure", Evidence{Message: "exit status 3", Detail: "panic: unreachable state", ExitCode: 3}, CauseProcessFailure, "nonzero-exit"},
		{"tool failure", Evidence{Message: "tool call failed: repograph returned malformed edges"}, CauseToolFailure, "tool-report"},
		{"unknown", Evidence{Message: "the operation ended unexpectedly"}, CauseUnknown, "unmatched"},
		{"empty evidence", Evidence{}, CauseUnknown, "unmatched"},
	}
	covered := map[Cause]bool{}
	for _, sample := range corpus {
		first := Normalize(sample.evidence)
		second := Normalize(sample.evidence)
		if first != second {
			t.Fatalf("%s: normalization is not deterministic: %+v then %+v", sample.name, first, second)
		}
		if first.Cause != sample.cause || first.Rule != sample.rule || first.ClassifierVersion != ClassifierVersion {
			t.Fatalf("%s: classification = %+v", sample.name, first)
		}
		if !ValidCause(first.Cause) {
			t.Fatalf("%s: classifier produced a foreign cause %q", sample.name, first.Cause)
		}
		covered[first.Cause] = true
	}
	for _, cause := range CanonicalCauses {
		if !covered[cause] {
			t.Fatalf("corpus never reaches canonical cause %q", cause)
		}
	}
	if ValidCause(Cause("crashed")) {
		t.Fatal("foreign cause accepted")
	}
}
