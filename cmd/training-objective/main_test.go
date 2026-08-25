package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

// TestObjectivePublishesTextToVideoPair pins the whole chain: a
// registered directory corpus, a published flow-matching text-to-video
// objective referencing it, a compiled objective program carrying the
// forward/backward/Muon phases, and a matrix that reports the pair
// trainable instead of refusing it for want of corpus and evidence.
func TestObjectivePublishesTextToVideoPair(t *testing.T) {
	ctx := context.Background()
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	corpus := t.TempDir()
	if err := os.WriteFile(filepath.Join(corpus, "clip.mp4"), make([]byte, 2048), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(corpus, "captions.json"), []byte(`[{"vid":"clip","caption":"c"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := dataset.RegisterDirectoryDataset(ctx, store, "clip-corpus", corpus); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := run([]string{
		"-repo", repository, "-name", "text to video flow matching",
		"-kind", string(trainingprogram.ObjectiveFlowMatching),
		"-input", "text", "-output", "video",
		"-metric", string(trainingprogram.MetricVideoPSNRSigned),
		"-dataset", "clip-corpus",
		"-evidence", "fixture: flow-matching objective over the registered clip corpus",
	}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "pair text->video") {
		t.Fatalf("publish output = %q", out.String())
	}
	if err := run([]string{
		"-repo", repository, "-name", "x", "-kind", string(trainingprogram.ObjectiveFlowMatching),
		"-input", "text", "-output", "video", "-metric", string(trainingprogram.MetricVideoPSNRSigned),
		"-dataset", "absent-corpus", "-evidence", "x",
	}, &out); err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("unregistered corpus accepted: %v", err)
	}

	store, err = overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objectiveID, found, err := store.ResolveAlias(ctx, "objective.registered.text-video")
	if err != nil || !found {
		t.Fatalf("objective alias = (%t, %v)", found, err)
	}
	document, err := trainingprogram.LoadObjective(ctx, store, objectiveID)
	if err != nil {
		t.Fatal(err)
	}
	if document.Kind != trainingprogram.ObjectiveFlowMatching || document.Authority != trainingprogram.ObjectiveDeclared {
		t.Fatalf("objective = %+v", document.ObjectiveSpec)
	}
	plan, err := optimizer.CompilePlan(4, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 4, Rows: 2, Cols: 2}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := trainingprogram.CompileObjectiveProgram(
		trainingprogram.ObjectiveFlowMatching,
		[]trainingprogram.ParameterSpec{{Name: "weight", Rows: 2, Cols: 2, Trainable: true}},
		plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := trainingprogram.CompileTrainingObjectiveMatrix(ctx, store,
		[]artifact.ID{objectiveID}, []trainingprogram.TrainingProgram{program})
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled) != 1 || compiled[0].Disposition != trainingprogram.ObjectiveProgramCompiled {
		t.Fatalf("compiled rows = %+v, want the flow-matching program bound", compiled)
	}
	matrix, err := trainingprogram.CompileObjectiveMatrix(ctx, store, []artifact.ID{objectiveID})
	if err != nil {
		t.Fatal(err)
	}
	trainable := false
	for _, row := range matrix {
		if row.Signature.Inputs[0] == recipecontract.ModalityText && row.Signature.Outputs[0] == recipecontract.ModalityVideo {
			trainable = row.Disposition == trainingprogram.ObjectiveTrainable
		}
	}
	if !trainable {
		t.Fatal("text->video still refused after publication")
	}
}
