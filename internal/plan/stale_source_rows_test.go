package plan

import (
	"strings"
	"testing"
)

// TestProspectiveMergeDropsCompletedIncomingRows pins the semantic-union
// rule for a source that still lists rows the target has since landed: the
// union drops an incoming step or item only when the combined authority
// proves its completion or retirement, and the target's own rows are never
// dropped.
func TestProspectiveMergeDropsCompletedIncomingRows(t *testing.T) {
	const repository = "test-repository"
	const localRevision = "1111111111111111111111111111111111111111"
	const incomingRevision = "2222222222222222222222222222222222222222"
	next := Item{
		ID: "next", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}
	landed := Item{
		ID: "landed", Status: StatusOpen,
		Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
	}
	split := Item{
		ID: "split", Status: StatusOpen,
		Steps: []Step{
			{ID: "first", Status: StatusOpen, Verify: "go test ./..."},
			{ID: "second", Status: StatusOpen, Verify: "go test ./..."},
		},
	}
	splitRemainder := Item{ID: "split", Status: StatusOpen, Steps: []Step{split.Steps[1]}}
	local := Plan{Items: []Item{next, splitRemainder}}
	incoming := Plan{Items: []Item{landed, next, split}}
	merged := local
	verify := func(localAuthority, incomingAuthority CompletionAuthority) error {
		return VerifyProspectiveMergeAuthorityWithProjection(
			repository, localRevision, incomingRevision,
			local, incoming, merged, localAuthority, incomingAuthority, MergeProjectionSemanticUnion,
		)
	}

	localAuthority := prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	incomingAuthority := prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	err := verify(localAuthority, incomingAuthority)
	if err == nil || !strings.Contains(err.Error(), "deleted incoming landed/do without gated completion evidence") {
		t.Fatalf("unproven incoming deletion error = %v", err)
	}

	localAuthority.completedReferences["landed/do"] = completionEvidence{commit: localRevision}
	err = verify(localAuthority, incomingAuthority)
	if err == nil || !strings.Contains(err.Error(), "deleted incoming split/first without gated completion evidence") {
		t.Fatalf("unproven incoming step deletion error = %v", err)
	}

	localAuthority.completedReferences["split/first"] = completionEvidence{commit: localRevision}
	if err := verify(localAuthority, incomingAuthority); err != nil {
		t.Fatalf("target-completed incoming rows refused: %v", err)
	}

	// Retirement of the whole item proves the deletion as well as completion.
	delete(localAuthority.completedReferences, "landed/do")
	localAuthority.retiredItems["landed"] = completionEvidence{commit: localRevision, retiredItem: true}
	if err := verify(localAuthority, incomingAuthority); err != nil {
		t.Fatalf("target-retired incoming item refused: %v", err)
	}

	// The incoming side's own evidence counts under the union.
	delete(localAuthority.retiredItems, "landed")
	incomingAuthority.completedReferences["landed/do"] = completionEvidence{commit: incomingRevision}
	if err := verify(localAuthority, incomingAuthority); err != nil {
		t.Fatalf("incoming-completed incoming row refused: %v", err)
	}

	// The target's own rows never leave a union, whatever the evidence.
	merged = Plan{Items: []Item{splitRemainder}}
	localAuthority.completedReferences["next/do"] = completionEvidence{commit: localRevision}
	err = verify(localAuthority, incomingAuthority)
	if err == nil || !strings.Contains(err.Error(), "deleted a parent item or step") {
		t.Fatalf("local deletion error = %v", err)
	}

	// First-parent-target keeps ignoring incoming identities entirely.
	merged = local
	incomingAuthority = prospectiveMergeAuthority(t, incoming, repository, incomingRevision, "common")
	localAuthority = prospectiveMergeAuthority(t, local, repository, localRevision, "common")
	if err := VerifyProspectiveMergeAuthorityWithProjection(
		repository, localRevision, incomingRevision,
		local, incoming, merged, localAuthority, incomingAuthority, MergeProjectionFirstParentTarget,
	); err != nil {
		t.Fatalf("first-parent-target projection refused: %v", err)
	}
}
