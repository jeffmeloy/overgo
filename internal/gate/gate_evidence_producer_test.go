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

// TestGateEvidenceNeedsTheGateProducer mints each kind only the gate writes
// through both doors: the plain one refuses it and the gate's capability
// admits it. The lifecycle is what the plan completion authority discovers a
// landing through. Gate results and attempts stay open to the plain door,
// because other verifiers mint them for their own runs; the row that decides
// their producers turns that assertion around.
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

	mint := func(schema string) artifact.Batch {
		contract := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: schema}
		content := mustGateValue(contract.ContentBytes(fmt.Appendf(nil, "{%q:true}", schema)))
		return artifact.Batch{Key: "mint/" + schema, Contents: []artifact.Content{content}}
	}
	for _, schema := range []string{
		runrecord.GateLifecycleSchema, runrecord.GateLaneObligationSchema, runrecord.SuiteCostSchema,
		packageReceiptSchema, batchEvidenceSchema, causeSchema,
	} {
		if _, err := store.Commit(ctx, mint(schema)); !errors.Is(err, overgodb.ErrProducerRefused) {
			t.Fatalf("the plain door minted %s: %v", schema, err)
		}
		if _, err := store.CommitAs(ctx, gateProducer, mint(schema)); err != nil {
			t.Fatalf("the gate's capability was refused %s: %v", schema, err)
		}
	}
	for _, shared := range []string{runrecord.GateSchema, runrecord.AttemptSchema} {
		if _, err := store.Commit(ctx, mint(shared)); err != nil {
			t.Fatalf("the shared kind %s was refused before its producers were decided: %v", shared, err)
		}
	}
}
