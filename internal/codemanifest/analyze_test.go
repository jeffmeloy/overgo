package codemanifest

import (
	"reflect"
	"testing"
)

// TestAnalyzeMatchesDiffAndClose holds the planner's one-pass analysis to
// exactly what the separate delta and closure give: validating each manifest
// once rather than in both steps changes the work, not the answer, and a
// manifest that fails validation is still refused.
func TestAnalyzeMatchesDiffAndClose(t *testing.T) {
	t.Parallel()
	base := completeFixtureManifest(t)
	candidateValue := fixtureManifest()
	candidateValue.Uncertainty = nil
	candidateValue.Symbols[1].BodySHA256 = "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	candidate, err := codec.New(candidateValue)
	if err != nil {
		t.Fatal(err)
	}
	wantDelta, err := Diff(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	wantImpact, err := Close(base, candidate, wantDelta)
	if err != nil {
		t.Fatal(err)
	}
	delta, impact, err := Analyze(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(delta, wantDelta) || !reflect.DeepEqual(impact, wantImpact) {
		t.Fatalf("one-pass analysis differs:\n%+v\n%+v\nwant\n%+v\n%+v", delta, impact, wantDelta, wantImpact)
	}
	candidate.SourceIdentity = base.SourceIdentity + "0"
	if _, _, err := Analyze(base, candidate); err == nil {
		t.Fatal("a manifest changed after identification was analyzed")
	}
}
