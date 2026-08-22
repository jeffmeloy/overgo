package plan

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestCompletionReceiptBindsGateEvidence(t *testing.T) {
	authority, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("authority"))
	gateResult, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("gate"))
	receipt, err := NewCompletionReceipt(
		"item", "step", "developer", authority,
		strings.Repeat("a", 40), gateResult,
	)
	wantLineage := []artifact.Lineage{
		{Child: receipt.ID, Parent: authority, Relation: artifact.RelationDependsOn},
		{Child: receipt.ID, Parent: gateResult, Relation: artifact.RelationDependsOn},
	}
	if err != nil || !receipt.ID.Valid() || !slices.Equal(receipt.Lineage(), wantLineage) {
		t.Fatalf("completion receipt = %+v, %v", receipt, err)
	}
	content, err := receipt.Content()
	if err != nil || content.Descriptor.ID != receipt.ID {
		t.Fatalf("completion content = %+v, %v", content.Descriptor, err)
	}
}

func TestMergeCompletionConflict(t *testing.T) {
	item := Item{ID: "item", Title: "item", Status: "open", Steps: []Step{{ID: "step", Title: "step", Status: "open", Verify: "go test ./..."}}}
	base := Plan{Campaign: "campaign", Doctrine: "doctrine", Items: []Item{item}}
	localAuthority, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("local"))
	upstreamAuthority, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("upstream"))
	local := Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Completed: []CompletionRef{{Item: "item", Step: "step", Authority: localAuthority}}}
	upstream := Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Completed: []CompletionRef{{Item: "item", Step: "step", Authority: upstreamAuthority}}}
	if _, err := MergeOpenProjections(base, local, upstream); err == nil {
		t.Fatal("conflicting completion authorities merged")
	}
}
