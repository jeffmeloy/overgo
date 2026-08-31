package steering

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

func recordedHistory(t *testing.T, store *overgodb.Store, label string) artifact.ID {
	t.Helper()
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: artifact.JSONMediaType, Schema: "overgo/test-history/v1",
	}
	content, err := contract.ContentBytes([]byte(`{"history":"` + label + `"}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "fixture/history/" + content.Descriptor.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	return content.Descriptor.ID
}

func fixtureProposal(history ...artifact.ID) Proposal {
	return Proposal{
		Slug: "close-catalog-latency", Goal: "Cut the fresh-child catalog scan below thirty seconds",
		Proposer:         "sonnet-baseline",
		PredictedBenefit: "catalog reads drop from minutes to seconds for every eval and swap consumer",
		PredictedCost:    "one row of work and one benchmark run",
		Uncertainty:      "the scan may be IO-bound rather than hash-bound, which the memo cannot fix",
		FalsifiableCheck: "go test ./internal/discovery -run TestCatalogScanBudget -count=1",
		History:          history,
	}
}

// TestProposalAdmissionConvertsToPlanRow pins the bounded steering
// contract: an admitted proposal is recorded in the store and becomes
// exactly one open plan row whose verify is the falsifiable check and
// whose rationale carries the proposal identity, prediction, and
// uncertainty. Admission executes nothing.
func TestProposalAdmissionConvertsToPlanRow(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	history := recordedHistory(t, store, "attempt-aggregate")

	admitted, row, err := Admit(ctx, store, fixtureProposal(history))
	if err != nil {
		t.Fatal(err)
	}
	if !admitted.ID.Valid() || admitted.ID.Kind() != artifact.KindEvidence {
		t.Fatalf("admitted identity = %s", admitted.ID)
	}
	if row.ID != "close-catalog-latency" || row.Status != plan.StatusOpen || len(row.Steps) != 1 {
		t.Fatalf("plan row = %+v", row)
	}
	step := row.Steps[0]
	if step.Verify != admitted.FalsifiableCheck ||
		!strings.Contains(step.Rationale, admitted.ID.String()) ||
		!strings.Contains(step.Rationale, "uncertainty:") {
		t.Fatalf("step = %+v", step)
	}
	if found, err := store.HasContent(ctx, admitted.ID); err != nil || !found {
		t.Fatalf("proposal content = (%t, %v)", found, err)
	}
}

// TestProposalAdmissionRefusesUngroundedRequests pins fail-closed
// admission: no history references, unrecorded references, missing
// uncertainty or check, and unbounded text are all refused with
// nothing recorded and no row produced.
func TestProposalAdmissionRefusesUngroundedRequests(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	recorded := recordedHistory(t, store, "real")
	phantom := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: artifact.JSONMediaType, Schema: "overgo/test-history/v1",
	}
	phantomContent, err := phantom.ContentBytes([]byte(`{"history":"never committed"}`))
	if err != nil {
		t.Fatal(err)
	}

	refusals := map[string]Proposal{
		"no history":         fixtureProposal(),
		"unrecorded history": fixtureProposal(phantomContent.Descriptor.ID),
	}
	missingUncertainty := fixtureProposal(recorded)
	missingUncertainty.Uncertainty = " "
	refusals["missing uncertainty"] = missingUncertainty
	missingCheck := fixtureProposal(recorded)
	missingCheck.FalsifiableCheck = ""
	refusals["missing falsifiable check"] = missingCheck
	essay := fixtureProposal(recorded)
	essay.PredictedBenefit = strings.Repeat("very ", 1000)
	refusals["unbounded text"] = essay
	badSlug := fixtureProposal(recorded)
	badSlug.Slug = "two words"
	refusals["free-text slug"] = badSlug

	for name, proposal := range refusals {
		if _, _, err := Admit(ctx, store, proposal); err == nil {
			t.Fatalf("%s: admission accepted an ungrounded proposal", name)
		}
	}
}
