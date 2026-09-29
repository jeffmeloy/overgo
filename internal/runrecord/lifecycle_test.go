package runrecord

import (
	"errors"
	"testing"
	"time"

	"overgo/internal/artifact"
)

func TestGateLifecycleConstruction(t *testing.T) {
	environment, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("lifecycle-environment"))
	if err != nil {
		t.Fatal(err)
	}
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("lifecycle-result"))
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := newGatePreparation(string(make([]byte, 64)), environment, time.Unix(100, 0))
	if err == nil {
		t.Fatal("accepted a non-hex tree key")
	}
	prepared, err = newGatePreparation("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", environment, time.Unix(100, 0))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := newGateFinalization(prepared, "0123456789abcdef0123456789abcdef01234567", result, OutcomeSucceeded); err != nil {
		t.Fatal(err)
	}
}

// newGatePreparation and newGateFinalization build the retired gate's
// lifecycle records, which the store still holds and the finalization
// lookup still reads; nothing writes new ones.
func newGatePreparation(treeKey string, environment artifact.ID, started time.Time) (gateLifecycle, error) {
	return gateLifecycleCodec.NewInitial(gateLifecycle{
		State: gatePrepared, TreeKey: treeKey,
		Environment: environment, Started: started.UTC().Format(time.RFC3339Nano),
	})
}

func newGateFinalization(preparation gateLifecycle, codeCommit string, result artifact.ID, outcome Outcome) (gateLifecycle, error) {
	if preparation.State != gatePrepared || preparation.ID.Kind() != artifact.KindEvidence {
		return gateLifecycle{}, errors.New("run record: finalization requires an identified preparation")
	}
	preparationID := preparation.ID
	resultID := result
	return gateLifecycleCodec.NewInitial(gateLifecycle{
		State: gateFinalized, TreeKey: preparation.TreeKey,
		Environment: preparation.Environment, Started: preparation.Started,
		Preparation: &preparationID, CodeCommit: codeCommit, Result: &resultID, Outcome: outcome,
	})
}
