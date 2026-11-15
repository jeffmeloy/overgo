package main

import (
	"bytes"
	"os/exec"
	"strings"
	"testing"
	"time"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheMatrixReadsEachClaimTestsLastPass holds a claim's standing to its test's last pass as the trunk's record keeps
// it (single-source/claim-record): a claim whose test passed at the head says so, one whose bound source moved after
// the pass says which file moved, and one whose test has no pass kept says that; none is refused. A claim had been
// rebound only by its test run again in a landing's fix stage, under an environment of its own.
func TestTheMatrixReadsEachClaimTestsLastPass(t *testing.T) {
	t.Parallel()
	root, _ := fixtureRepository(t)
	source := writeEvidence(t, root, "feature/feature.go", "package feature\n\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "feature/feature_test.go", "package feature\n\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	other := writeEvidence(t, root, "feature/other_test.go", "package feature\n\nfunc TestOther() {}\n", "func TestOther(", roleArtifact)
	writeTestManifest(t, root, []claim{
		{ID: "feature", EvidenceTier: tierContract, Summary: "Feature works.", Evidence: []evidence{source, proof}},
		{ID: "other", EvidenceTier: tierContract, Summary: "Other works.", Evidence: []evidence{source, other}},
	}, testModels())
	passed := commitFixture(t, root, "claims")
	testcheck.NoError(t, plan.RecordTestPasses(root, plan.TestPass{Commit: passed, Ended: time.Now()}, []string{plan.PassedTest("overgo/feature", "TestFeature")}))
	standing := func() string {
		var out bytes.Buffer
		testcheck.NoError(t, claimStandings(t.Context(), &out, root))
		return out.String()
	}
	if got := standing(); !strings.Contains(got, "claim feature: TestFeature passed at "+passed[:12]) || strings.Contains(got, "moved") ||
		!strings.Contains(got, "claim other: TestOther has no pass kept") {
		t.Fatalf("the claims read before any source moved:\n%s", got)
	}
	testutil.WriteTextFile(t, root, "feature/feature.go", "package feature\n\nfunc Feature() int { return 1 }\n")
	commitFixture(t, root, "move the source")
	if got := standing(); !strings.Contains(got, "claim feature: TestFeature passed at "+passed[:12]+", and feature/feature.go moved since") {
		t.Fatalf("the claims read after the source moved:\n%s", got)
	}
}

// commitFixture commits the fixture repository's tree and returns the commit.
func commitFixture(t *testing.T, root, message string) string {
	t.Helper()
	var head string
	for _, arguments := range [][]string{{"add", "-A"}, {"commit", "-q", "-m", message}, {"rev-parse", "HEAD"}} {
		command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...)
		command.Dir = root
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
		head = strings.TrimSpace(string(output))
	}
	return head
}
