package gate

import (
	"errors"
	"fmt"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// mintKind builds a batch that introduces one document of schema, depending
// on parents: what a caller holding the store API can always construct.
func mintKind(t *testing.T, schema, label string, parents ...artifact.ID) (artifact.Batch, artifact.ID) {
	t.Helper()
	contract := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: schema}
	content := mustGateValue(contract.ContentBytes(fmt.Appendf(nil, "{%q:%q}", schema, label)))
	id := content.Descriptor.ID
	return artifact.Batch{
		Key: "mint/" + id.String(), Contents: []artifact.Content{content}, Lineage: artifact.DependencyLineage(id, parents...),
	}, id
}

// TestGateEvidenceNeedsTheGateProducer mints each kind only the gate writes
// through both doors: the plain one refuses it and the gate's capability
// admits it. The lifecycle is what the plan completion authority discovers a
// landing through.
func TestGateEvidenceNeedsTheGateProducer(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := mustGateValue(overgodb.Open(filepath.Join(t.TempDir(), "store")))
	defer store.Close()
	// The selection cause keeps its schema inside its codec: read it off a
	// real document, so a renamed kind cannot slip past a literal here.
	cause := mustGateValue(runrecord.SelectionCauseCodec.NewInitial(runrecord.SelectionCauseRecord{
		Result:   testutil.ArtifactID(t, artifact.KindEvidence, "gate result"),
		Changed:  []string{"internal/reader/reader.go"},
		Packages: []runrecord.SelectionPackage{{Package: "overgo/internal/reader", Step: "test-owners", Action: "pass", Started: true}},
	}))
	causeSchema := mustGateValue(runrecord.SelectionCauseCodec.Content(cause)).Descriptor.Schema

	for _, schema := range []string{
		runrecord.GateLifecycleSchema, runrecord.GateLaneObligationSchema, runrecord.SuiteCostSchema, runrecord.AttemptSchema,
		packageReceiptSchema, batchEvidenceSchema, causeSchema,
	} {
		batch, _ := mintKind(t, schema, "minted")
		if _, err := store.Commit(ctx, batch); !errors.Is(err, overgodb.ErrProducerRefused) {
			t.Fatalf("the plain door minted %s: %v", schema, err)
		}
		if _, err := store.CommitAs(ctx, gateProducer, batch); err != nil {
			t.Fatalf("the gate's capability was refused %s: %v", schema, err)
		}
	}
}

// TestPlanAttemptsAreTheGatesAlone closes the path by which a caller holding
// the API could wedge the plan completion authority, which requires exactly
// one successful attempt under a gate result: a second attempt hung off a
// real result is refused by the store and leaves the head where it was, while
// the gate's capability may write it. The result itself stays a shared kind
// -- other verifiers mint results for their own runs -- because the authority
// accepts one only through a gate-only finalization introduced with it.
func TestPlanAttemptsAreTheGatesAlone(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := mustGateValue(overgodb.Open(filepath.Join(t.TempDir(), "store")))
	defer store.Close()
	verifierRun, result := mintKind(t, runrecord.GateSchema, "another verifier's run")
	if _, err := store.Commit(ctx, verifierRun); err != nil {
		t.Fatalf("a verifier outside the gate could not record its own result: %v", err)
	}
	forged, _ := mintKind(t, runrecord.AttemptSchema, "a second successful attempt", result)
	_, before := store.Head()
	if _, err := store.Commit(ctx, forged); !errors.Is(err, overgodb.ErrProducerRefused) {
		t.Fatalf("an attempt was hung off a gate result through the plain door: %v", err)
	}
	if _, after := store.Head(); after != before {
		t.Fatalf("a refused attempt moved the head from %d to %d", before, after)
	}
	if _, err := store.CommitAs(ctx, gateProducer, forged); err != nil {
		t.Fatalf("the gate's capability was refused its own attempt: %v", err)
	}
}
