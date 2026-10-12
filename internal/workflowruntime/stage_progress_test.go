package workflowruntime

import (
	"context"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// commitWatch is a repository that shows a test each batch once the store
// under it has accepted it. A stage's receipt becomes durable in Commit and
// nowhere else, so a test that must act when a receipt is durable waits here
// and not on anything the runtime says about itself.
type commitWatch struct {
	artifact.Repository
	committed func(artifact.Batch)
}

// Commit passes the batch to the store and, when the store took it, to the test.
func (store *commitWatch) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	id, err := store.Repository.Commit(ctx, batch)
	if err == nil {
		store.committed(batch)
	}
	return id, err
}

// TestStageProgressPrecedesSiblingCompletion holds that a completed stage's
// receipt is durable while a sibling stage of the same ready set is still
// running. The right branch is held until the store has accepted the left
// branch's completed receipt, and that receipt must then be the one the left
// stage resolves to. A runtime that kept receipts back until the whole set had
// finished would never release the right branch; the test runner owns the time
// that takes.
func TestStageProgressPrecedesSiblingCompletion(t *testing.T) {
	fixture := newParallelFixture(t)
	ctx := t.Context()
	release := make(chan struct{})
	watched := &commitWatch{Repository: fixture.store, committed: func(batch artifact.Batch) {
		for _, content := range batch.Contents {
			if content.Descriptor.Schema != runrecord.StageReceiptSchema {
				continue
			}
			receipt, err := runrecord.ParseStageReceipt(content.Data)
			if err != nil {
				t.Error(err)
				continue
			}
			if receipt.Node != "left" || receipt.State != runrecord.StageCompleted {
				continue
			}
			stored, found, err := runrecord.ResolveStageReceipt(ctx, fixture.store, fixture.operation, "left")
			if err != nil || !found || stored.ID != receipt.ID || stored.State != runrecord.StageCompleted {
				t.Errorf("left stage resolves to (%+v, %t, %v) after the store accepted its completed receipt %s",
					stored, found, err, receipt.ID)
			}
			close(release)
		}
	}}
	var err error
	if fixture.runtime, err = NewForProgram(watched, fixture.program); err != nil {
		t.Fatal(err)
	}
	registerParallelAdapters(t, fixture, func(ctx context.Context, value string) (string, error) {
		if value == "right" {
			select {
			case <-release:
			case <-ctx.Done():
				return "", context.Cause(ctx)
			}
		}
		return value, nil
	})
	if _, err := fixture.execute(ctx); err != nil {
		t.Fatal(err)
	}
}

// TestRuntimeHasNoStageObserver holds that a runtime offers no callback for
// completed stages: no ObserveStages method, and no field that could keep a
// receipt callback. The document speech graph was the callback's only caller
// and is retired, so a second way to learn of a stage's progress would have
// nothing in production to keep it honest. Progress is read from the store.
func TestRuntimeHasNoStageObserver(t *testing.T) {
	runtimeType := reflect.TypeFor[*Runtime]()
	if _, found := runtimeType.MethodByName("ObserveStages"); found {
		t.Error("*Runtime still has the method ObserveStages")
	}
	callback := reflect.TypeFor[func(runrecord.StageReceipt)]()
	for field := range runtimeType.Elem().Fields() {
		if field.Type == callback {
			t.Errorf("Runtime still keeps a stage receipt callback in its field %s", field.Name)
		}
	}
}
