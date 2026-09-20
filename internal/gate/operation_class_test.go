package gate

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// TestOperationClassIsPrinted holds the class note to the audit printer: the
// note the precheck records is the line the operator reads, so a class the
// gate derived cannot stay invisible behind the printer's prefix filter.
func TestOperationClassIsPrinted(t *testing.T) {
	t.Parallel()
	store := mustGateValue(overgodb.Open(filepath.Join(t.TempDir(), "store")))
	defer store.Close()
	g := &gateContext{paths: []string{"internal/gate/gate.go"}}
	if err := g.requireOperationPrechecks(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	if printed := compactAudit(g.audit); len(printed) != 1 || printed[0] != "advisory: class: operation class: "+classCodeChange {
		t.Fatalf("printed audit = %q", printed)
	}
}

// TestOperationClassPrechecks derives a landing's classes from the ship set
// and the store's chain, and holds a store mutation to the precheck its
// class owns: content released since the last landing refuses the gate until
// a passed proof follows it, a proof typed by hand is not admitted by the
// store at all, a failed proof proves nothing, and a release newer than the
// proof puts the store back in debt.
func TestOperationClassPrechecks(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		paths            []string
		released, landed uint64
		want             string
	}{
		{[]string{plan.Path}, 0, 0, classPlanEdit},
		{[]string{plan.Path, "internal/gate/run.go"}, 4, 9, classCodeChange},
		{[]string{plan.Path}, 9, 4, classPlanEdit + "+" + classStoreMutation},
	} {
		if got := strings.Join(operationClasses(test.paths, test.released, test.landed), "+"); got != test.want {
			t.Fatalf("classes of %v released=%d landed=%d = %q, want %q", test.paths, test.released, test.landed, got, test.want)
		}
	}

	ctx := t.Context()
	store := mustGateValue(overgodb.Open(filepath.Join(t.TempDir(), "store")))
	defer store.Close()
	cache := artifact.DocumentContract{Kind: artifact.KindOutput, MediaType: "application/json", Schema: "test/derived-cache/v1"}
	release := func(label string) {
		t.Helper()
		content := mustGateValue(cache.ContentBytes(fmt.Appendf(nil, "{%q:true}", label)))
		mustGateValue(store.Commit(ctx, artifact.Batch{Key: "cache/" + label, Contents: []artifact.Content{content}}))
		report := mustGateValue(overgodb.Release(ctx, store, func(string, artifact.ID) error { return nil },
			overgodb.RetentionPolicy{}, func(descriptor artifact.Descriptor) bool { return descriptor.Schema == cache.Schema }))
		if report.Released != 1 || store.LastRelease() != report.SequenceAfter {
			t.Fatalf("release of %s = %+v, last release %d", label, report, store.LastRelease())
		}
	}
	prove := func(passed bool) artifact.Batch {
		t.Helper()
		proof := overgodb.OperationProof{Backup: "sealed", Checks: []overgodb.ConsumerCheck{{Name: "completion-authority", Passed: passed}}, Passed: passed}
		return mustGateValue(proof.Batch(ctx, store))
	}
	g := &gateContext{paths: []string{plan.Path}}
	refused := func(state string) {
		t.Helper()
		if err := g.requireOperationPrechecks(ctx, store); err == nil || !strings.Contains(err.Error(), "store-precheck") {
			t.Fatalf("%s admitted the gate: %v", state, err)
		}
	}

	if err := g.requireOperationPrechecks(ctx, store); err != nil || g.audit[0] != "operation class: "+classPlanEdit {
		t.Fatalf("an unmutated store = %v, audit %v", err, g.audit)
	}
	release("first")
	refused("a release with no proof")
	if last := g.audit[len(g.audit)-1]; !strings.HasSuffix(last, "+"+classStoreMutation) {
		t.Fatalf("a released store was classed %q", last)
	}
	if _, err := store.Commit(ctx, prove(true)); !errors.Is(err, overgodb.ErrProducerRefused) {
		t.Fatalf("a proof typed by hand was admitted: %v", err)
	}
	precheck := overgodb.NewProducer("store-precheck")
	mustGateValue(store.CommitAs(ctx, precheck, prove(false)))
	refused("a failed proof")
	mustGateValue(store.CommitAs(ctx, precheck, prove(true)))
	if err := g.requireOperationPrechecks(ctx, store); err != nil {
		t.Fatalf("a proven store mutation was refused: %v", err)
	}
	release("second")
	refused("a release newer than the proof")
}
