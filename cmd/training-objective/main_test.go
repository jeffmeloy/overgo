package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
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
	ctx := t.Context()
	repository := registerClipCorpus(t)
	var out bytes.Buffer
	if err := run(clipObjectiveArgs(repository, "clip-corpus", "record", "train=1,heldout=1"), &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "pair text->video") {
		t.Fatalf("publish output = %q", out.String())
	}
	if err := run(clipObjectiveArgs(repository, "absent-corpus", "record", "train=1,heldout=1"), &out); err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("unregistered corpus accepted: %v", err)
	}

	store, err := overgodb.OpenReadOnly(repository)
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

// TestPublishObjectiveBindsTrainingMembership holds publication to a real
// split: the objective's Split is the committed train membership of a group
// split of the registered dataset, disjoint from its held-out partition, so
// a held-out view can be compiled for it. The grouping and weights are
// required, and a split that would leave a partition empty is refused.
func TestPublishObjectiveBindsTrainingMembership(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	repository := registerClipCorpus(t)
	for _, refused := range []struct{ group, weights, want string }{
		{"", "train=1,heldout=1", "-split-group is required"},
		{"record", "train=1", "both train and heldout are required"},
		{"row", "train=1,heldout=1", "want asset or record"},
		{"asset", "train=1,heldout=1", "leaves"},
	} {
		if err := run(clipObjectiveArgs(repository, "clip-corpus", refused.group, refused.weights), io.Discard); err == nil || !strings.Contains(err.Error(), refused.want) {
			t.Fatalf("split %q %q = %v, want %q", refused.group, refused.weights, err, refused.want)
		}
	}
	if err := run(clipObjectiveArgs(repository, "clip-corpus", "record", "train=3,heldout=1"), io.Discard); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objectiveID, found, err := store.ResolveAlias(ctx, "objective.registered.text-video")
	if err != nil || !found {
		t.Fatalf("objective alias = (%t, %v)", found, err)
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, objectiveID)
	if err != nil {
		t.Fatal(err)
	}
	training, found, err := dataset.LoadMembership(ctx, store, objective.Split)
	if err != nil || !found || training.Partition != trainPartition || training.Source != objective.Dataset {
		t.Fatalf("objective split = (%+v, %t, %v), want the committed train membership of its dataset", training, found, err)
	}
	var heldout dataset.Membership
	for _, child := range mustChildren(t, store, objective.Dataset) {
		if membership, found, err := dataset.LoadMembership(ctx, store, child); err == nil && found && membership.Partition == heldoutPartition {
			heldout = membership
		}
	}
	if len(heldout.Records) == 0 || len(training.Records)+len(heldout.Records) != clipCorpusFiles {
		t.Fatalf("partitions hold %d train and %d held-out records, want all %d files split", len(training.Records), len(heldout.Records), clipCorpusFiles)
	}
	if _, err := evaluation.CompileSFTEvaluationView(objective, training, heldout); err != nil {
		t.Fatalf("a held-out view does not compile for the published objective: %v", err)
	}
}

// clipCorpusFiles is the clip corpus's file count: enough records that a
// declared split leaves neither partition empty.
const clipCorpusFiles = 12

// registerClipCorpus registers a directory of clips and captions as the
// clip-corpus dataset in a fresh store and returns the store's root.
func registerClipCorpus(t *testing.T) string {
	t.Helper()
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	corpus := t.TempDir()
	for index := range clipCorpusFiles - 1 {
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("clip-%02d.mp4", index)), make([]byte, 2048), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(corpus, "captions.json"), []byte(`[{"vid":"clip-00","caption":"c"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := dataset.RegisterDirectoryDataset(t.Context(), store, "clip-corpus", corpus); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return repository
}

// clipObjectiveArgs publishes a text-to-video flow-matching objective over
// datasetName with the given split declaration; an empty group or weights
// omits that flag.
func clipObjectiveArgs(repository, datasetName, group, weights string) []string {
	args := []string{
		"-repo", repository, "-name", "text to video flow matching",
		"-kind", string(trainingprogram.ObjectiveFlowMatching),
		"-input", "text", "-output", "video",
		"-metric", string(trainingprogram.MetricVideoPSNRSigned),
		"-dataset", datasetName,
		"-evidence", "fixture: flow-matching objective over the registered clip corpus",
	}
	if group != "" {
		args = append(args, "-split-group", group)
	}
	if weights != "" {
		args = append(args, "-split-weights", weights)
	}
	return args
}

func mustChildren(t *testing.T, store *overgodb.Store, id artifact.ID) []artifact.ID {
	t.Helper()
	lineage, err := store.Children(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	var children []artifact.ID
	for _, edge := range lineage {
		children = append(children, edge.Child)
	}
	return children
}
