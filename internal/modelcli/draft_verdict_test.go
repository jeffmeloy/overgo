package modelcli

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestDraftJudgementLeavesAHostRunnerAlone opens a host model as a server does and
// by an explicit judgement. It cannot draft, so neither open asks for a device
// identity it has no device to answer with.
func TestDraftJudgementLeavesAHostRunnerAlone(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	storeRoot, path := filepath.Join(root, "store"), filepath.Join(root, "model.gguf")
	writeServingFixture(t, path, 0)
	store, err := overgodb.Create(storeRoot)
	testcheck.NoError(t, err)
	err = servingtest.PublishActiveGGUFWithPolicy(t.Context(), store, path, recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference)
	testcheck.NoError(t, err)
	testcheck.NoError(t, store.Close())
	for name, opened := range map[string]func(context.Context) (*inference.Runner, error){
		"served": func(ctx context.Context) (*inference.Runner, error) {
			return OpenServedRunner(ctx, storeRoot, path, inference.OpenOptions{})
		},
		"judged": func(ctx context.Context) (*inference.Runner, error) {
			return OpenRunner(ctx, storeRoot, path, inference.OpenOptions{JudgeOwnDrafts: true})
		},
	} {
		runner, err := opened(t.Context())
		if err != nil {
			t.Fatalf("%s open: %v", name, err)
		}
		testcheck.NoError(t, runner.Close())
	}
}

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
		store := testcheck.Must(overgodb.OpenReadOnly(repository))(t)
		defer store.Close()
		return readDraftVerdicts(ctx, store)
	}
	testcheck.NoError(t, publishDraftVerdict(ctx, repository, first))
	testcheck.NoError(t, publishDraftVerdict(ctx, repository, second))
	first.Refused = false
	testcheck.NoError(t, publishDraftVerdict(ctx, repository, first))
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
