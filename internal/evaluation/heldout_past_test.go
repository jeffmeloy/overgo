package evaluation

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
)

// TestPastHeldoutVerdictsAreReadAsTheStoreStood judges two trained models on
// one objective, one after the other, and reads the verdicts back as a
// proposer may be shown them. Read through the sequence the journal stood at
// between the two, only the first is named, with its document's bytes; read as
// the store stands, both are, oldest first. A byte budget the verdicts named
// exceed refuses the read whole where it would have had to drop one, and a
// sequence the journal has not reached is refused.
func TestPastHeldoutVerdictsAreReadAsTheStoreStood(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	forecast := newHeldoutFixture(t, trainingprogram.MetricForecastMAE)
	commit := func(key string, stage func(string) (artifact.Batch, error), declared ...artifact.ID) {
		t.Helper()
		batch, err := stage(key)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range declared {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
		}
		if _, err := artifact.Publish(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	commit("test-split", func(key string) (artifact.Batch, error) { return forecast.split.PublicationBatch(key, nil) }, forecast.objective.Dataset)
	commit("test-objective", func(key string) (artifact.Batch, error) {
		content, err := forecast.objective.Content()
		return artifact.Batch{Key: key, Contents: []artifact.Content{content}}, err
	})
	session := heldoutTraining(t, store, forecast.objective.Dataset, forecast.view.TrainingMembership)
	base := forecast.score(t, 2, "falling")
	commit("test-view", forecast.view.Batch, forecast.objective.Processors...)
	commit("test-plan", forecast.plan.Batch)
	commit("test-base", base.Batch)
	// judge commits a trained model's report and its verdict, and returns the
	// verdict with the bytes of its document.
	judge := func(key string, offset float64) (artifact.ID, uint64) {
		t.Helper()
		candidate := forecast.score(t, offset, "falling")
		commit(key+"-report", candidate.Batch)
		verdict, err := JudgeHeldout(forecast.objective, forecast.view, forecast.plan, base, candidate, session)
		if err != nil {
			t.Fatal(err)
		}
		commit(key, verdict.Batch)
		content, err := verdict.Content()
		if err != nil {
			t.Fatal(err)
		}
		return verdict.ID, content.Descriptor.Size
	}
	first, firstBytes := judge("first", 1)
	_, between := store.Head()
	second, secondBytes := judge("second", 0.5)
	_, now := store.Head()

	read := func(through, budget uint64) ([]artifact.ID, uint64) {
		t.Helper()
		verdicts, bytes, err := PastHeldoutVerdicts(ctx, store, through, budget)
		if err != nil {
			t.Fatalf("verdicts through sequence %d under %d bytes: %v", through, budget, err)
		}
		return verdicts, bytes
	}
	if verdicts, bytes := read(between, 0); !slices.Equal(verdicts, []artifact.ID{first}) || bytes != firstBytes {
		t.Fatalf("through the sequence between the two verdicts the store names %v in %d bytes, want the first alone in %d", verdicts, bytes, firstBytes)
	}
	for _, through := range []uint64{0, now} {
		if verdicts, bytes := read(through, 0); !slices.Equal(verdicts, []artifact.ID{first, second}) || bytes != firstBytes+secondBytes {
			t.Fatalf("through sequence %d the store names %v in %d bytes, want both verdicts, oldest first, in %d", through, verdicts, bytes, firstBytes+secondBytes)
		}
	}
	// A budget the first verdict fills admits what the store held between the
	// two, and refuses the store as it stands instead of naming one of two.
	if verdicts, _ := read(between, firstBytes); !slices.Equal(verdicts, []artifact.ID{first}) {
		t.Fatalf("a budget of the first verdict's bytes names %v at the time it stood alone", verdicts)
	}
	if verdicts, _, err := PastHeldoutVerdicts(ctx, store, 0, firstBytes); err == nil {
		t.Fatalf("a budget of one verdict's bytes named %v of two", verdicts)
	}
	if verdicts, _, err := PastHeldoutVerdicts(ctx, store, now+1, 0); err == nil {
		t.Fatalf("a sequence the journal has not reached named %v", verdicts)
	}
	if facts, err := ReadHeldoutVerdict(ctx, store, first); err != nil || facts.BaselineValue != 2 || facts.CandidateValue != 1 {
		t.Fatalf("the first verdict reads %+v (%v), want the base at 2 and the trained model at 1", facts, err)
	}
}
