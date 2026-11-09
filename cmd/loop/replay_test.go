package main

import "testing"

// TestAReplayTakesStepsCheckedByOneTest holds which landed steps a replay takes: those whose check runs one Go test,
// named whole, in one package; a check of several tests, of a pattern, of two packages or of no test is passed over.
func TestAReplayTakesStepsCheckedByOneTest(t *testing.T) {
	t.Parallel()
	for verify, taken := range map[string]bool{
		"go test ./internal/hfbpe -count=1 -run '^TestAnEmptyAddedTokenIsRefused$'":                                     true,
		"OVERGO_CUDA_TEST=1 go test -tags modeltest ./cmd/loop -count=1 -run '^TestALocalModelWorksAFixtureStep$'":      true,
		"go test ./internal/a -count=1 -run '^TestOne$|^TestTwo$'":                                                      false,
		"go test ./internal/a -count=1 -run 'TestSome'":                                                                 false,
		"go test ./internal/a ./internal/b -count=1 -run '^TestOne$'":                                                   false,
		"go test ./internal/a -count=1 -run '^TestOne$' && go test ./internal/b -count=1 -run '^TestTwo$'":              false,
		"go run ./cmd/validate -scorecard":                                                                              false,
		"go test ./internal/a -count=1 -run '^TestOne$/^case$'":                                                         false,
		"go test ./cmd/plan -count=1 -run '^TestNoteAppendsToARationale$' && ! grep -q 'func TestX' cmd/plan/x_test.go": true,
	} {
		if got := oneTest(verify); got != taken {
			t.Errorf("oneTest(%q) = %t, want %t", verify, got, taken)
		}
	}
}
