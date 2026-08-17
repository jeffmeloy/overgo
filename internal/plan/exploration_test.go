package plan

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestExplorationBudgetTokensEnforced pins the bounded-exploration contract:
// GPU-minute grants are immutable tokens issued by an external authority;
// consumption derives from committed charges; spend within balance admits,
// exceeding the balance or any spend while contained refuses locally and
// escalates to an external decision; corrupt charge sets error rather than
// round away.
func TestExplorationBudgetTokensEnforced(t *testing.T) {
	proposer := testutil.ArtifactID(t, artifact.KindEvidence, "autonomous-proposer")
	authority := testutil.ArtifactID(t, artifact.KindEvidence, "external-authority")
	experiment := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindEvidence, name)
	}

	grant, err := NewExplorationGrant(proposer, authority, 120)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := NewExplorationGrant(proposer, authority, 120)
	if err != nil || replay.ID != grant.ID {
		t.Fatalf("replayed grant = (%v, %v), want identical identity", replay.ID, err)
	}
	if _, err := NewExplorationGrant(proposer, proposer, 120); err == nil {
		t.Fatal("self-issued grant accepted")
	}
	if _, err := NewExplorationGrant(proposer, authority, 0); err == nil {
		t.Fatal("empty grant accepted")
	}

	first, err := NewExplorationCharge(grant.ID, experiment("probe-1"), 90)
	if err != nil {
		t.Fatal(err)
	}
	admission, err := AdmitExploration(grant, []ExplorationCharge{first}, 20, false)
	if err != nil || !admission.Admitted || admission.RemainingGPUMinutes != 10 {
		t.Fatalf("within-balance admission = (%+v, %v)", admission, err)
	}
	admission, err = AdmitExploration(grant, []ExplorationCharge{first}, 40, false)
	if err != nil || admission.Admitted || !admission.RequiresExternalDecision {
		t.Fatalf("over-balance admission = (%+v, %v), want local refusal escalated externally", admission, err)
	}
	if !strings.Contains(admission.Reason, "external decision") {
		t.Fatalf("escalation reason = %q", admission.Reason)
	}
	admission, err = AdmitExploration(grant, []ExplorationCharge{first}, 5, true)
	if err != nil || admission.Admitted || !admission.RequiresExternalDecision {
		t.Fatalf("contained admission = (%+v, %v), want refusal", admission, err)
	}

	second, err := NewExplorationCharge(grant.ID, experiment("probe-2"), 40)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := AdmitExploration(grant, []ExplorationCharge{first, second}, 1, false); err == nil {
		t.Fatal("charges exceeding the grant folded silently")
	}
	otherGrant, err := NewExplorationGrant(authority, proposer, 60)
	if err != nil {
		t.Fatal(err)
	}
	stray, err := NewExplorationCharge(otherGrant.ID, experiment("stray"), 10)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := AdmitExploration(grant, []ExplorationCharge{stray}, 1, false); err == nil {
		t.Fatal("charge against another grant accepted")
	}
	if _, err := AdmitExploration(grant, nil, 0, false); err == nil {
		t.Fatal("zero-minute request accepted")
	}

	remaining, err := ExplorationBalance(grant, []ExplorationCharge{first})
	if err != nil || remaining != 30 {
		t.Fatalf("balance = (%d, %v), want 30", remaining, err)
	}
	content, err := grant.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseExplorationGrant(content.Data)
	if err != nil || parsed.ID != grant.ID || parsed.IssuedGPUMinutes != 120 {
		t.Fatalf("grant roundtrip = (%+v, %v)", parsed, err)
	}
	chargeContent, err := first.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsedCharge, err := ParseExplorationCharge(chargeContent.Data)
	if err != nil || parsedCharge.ID != first.ID || parsedCharge.GPUMinutes != 90 {
		t.Fatalf("charge roundtrip = (%+v, %v)", parsedCharge, err)
	}
}
