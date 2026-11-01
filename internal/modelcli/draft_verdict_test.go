package modelcli

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestADraftVerdictIsRecordedPerModelBuildAndCard holds the record of served models' draft verdicts to one verdict a
// model, build and card: a verdict on another card is kept beside the first, a verdict reached again replaces its own,
// and a subject nothing recorded finds none, so its open judges.
func TestADraftVerdictIsRecordedPerModelBuildAndCard(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	repository := t.TempDir()
	model := testutil.ArtifactID(t, artifact.KindModel, "Qwen3.5-4B")
	first := draftVerdict{Model: model, Execution: "build-a", Device: "GPU-1", Refused: true}
	second := draftVerdict{Model: model, Execution: "build-a", Device: "GPU-2", Refused: false}
	read := func() draftVerdicts {
		store, err := overgodb.OpenReadOnly(repository)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		return readDraftVerdicts(ctx, store)
	}
	if err := publishDraftVerdict(ctx, repository, first); err != nil {
		t.Fatal(err)
	}
	if err := publishDraftVerdict(ctx, repository, second); err != nil {
		t.Fatal(err)
	}
	first.Refused = false
	if err := publishDraftVerdict(ctx, repository, first); err != nil {
		t.Fatal(err)
	}
	record := read()
	if len(record.Rows) != 2 {
		t.Fatalf("the record holds %d verdicts, want one a card", len(record.Rows))
	}
	for _, subject := range []draftVerdict{first, second} {
		if verdict := record.recorded(subject); verdict == nil || *verdict != subject.Refused {
			t.Errorf("%s on %s reads %v, want refused=%t", subject.Execution, subject.Device, verdict, subject.Refused)
		}
	}
	if verdict := record.recorded(draftVerdict{Model: model, Execution: "build-b", Device: "GPU-1"}); verdict != nil {
		t.Errorf("another build reads the verdict %t, want none so its open judges", *verdict)
	}
}
