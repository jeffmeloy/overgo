package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/trainingdata"
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
	testcheck.NoError(t, run(clipObjectiveArgs(repository, "clip-corpus", "record", "train=1,heldout=1"), &out))
	if !strings.Contains(out.String(), "pair text->video") {
		t.Fatalf("publish output = %q", out.String())
	}
	if err := run(clipObjectiveArgs(repository, "absent-corpus", "record", "train=1,heldout=1"), &out); err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("unregistered corpus accepted: %v", err)
	}

	store := overgodbtest.OpenReadOnly(t, repository)
	objectiveID, found, err := store.ResolveAlias(ctx, "objective.registered.text-video")
	if err != nil || !found {
		t.Fatalf("objective alias = (%t, %v)", found, err)
	}
	document, err := trainingprogram.LoadObjective(ctx, store, objectiveID)
	testcheck.NoError(t, err)
	if document.Kind != trainingprogram.ObjectiveFlowMatching || document.Authority != trainingprogram.ObjectiveDeclared {
		t.Fatalf("objective = %+v", document.ObjectiveSpec)
	}
	plan, err := optimizer.CompilePlan(4, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 4, Rows: 2, Cols: 2}})
	testcheck.NoError(t, err)
	program, err := trainingprogram.CompileObjectiveProgram(
		trainingprogram.ObjectiveFlowMatching,
		[]trainingprogram.ParameterSpec{{Name: "weight", Rows: 2, Cols: 2, Trainable: true}},
		plan,
	)
	testcheck.NoError(t, err)
	compiled, err := trainingprogram.CompileTrainingObjectiveMatrix(ctx, store,
		[]artifact.ID{objectiveID}, []trainingprogram.TrainingProgram{program})
	testcheck.NoError(t, err)
	if len(compiled) != 1 || compiled[0].Disposition != trainingprogram.ObjectiveProgramCompiled {
		t.Fatalf("compiled rows = %+v, want the flow-matching program bound", compiled)
	}
	matrix, err := trainingprogram.CompileObjectiveMatrix(ctx, store, []artifact.ID{objectiveID})
	testcheck.NoError(t, err)
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

// TestObjectiveMatrixListsRegisteredPairs holds -matrix to the objectives the store registers.
//   - each pair with a registered objective is listed with the identity its alias binds
//   - every other pair of the six modalities is listed once as without an objective
func TestObjectiveMatrixListsRegisteredPairs(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	repository := registerClipCorpus(t)
	video := clipObjectiveArgs(repository, "clip-corpus", "record", "train=1,heldout=1")
	image := append(slices.Clone(video), "-name", "text to image flow matching",
		"-output", "image", "-metric", string(trainingprogram.MetricImagePSNRSigned))
	for _, args := range [][]string{video, image} {
		testcheck.NoError(t, run(args, io.Discard))
	}
	var out bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-matrix"}, &out))
	store := overgodbtest.OpenReadOnly(t, repository)
	var want []string
	for _, output := range []recipecontract.Modality{recipecontract.ModalityImage, recipecontract.ModalityVideo} {
		id, found, err := store.ResolveAlias(ctx, registeredObjectivePrefix+"text-"+string(output))
		if err != nil || !found {
			t.Fatalf("objective alias of text-%s = (%t, %v)", output, found, err)
		}
		want = append(want, fmt.Sprintf("pair text->%s objective %s", output, id))
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != len(want)+1 || !slices.Equal(lines[:len(want)], want) {
		t.Fatalf("listing = %q, want %q and then the pairs without an objective", lines, want)
	}
	count, pairs, found := strings.Cut(lines[len(want)], " pairs without an objective: ")
	without := strings.Fields(pairs)
	if !found || count != fmt.Sprint(len(without)) {
		t.Fatalf("last line = %q, want the count and the pairs without an objective", lines[len(want)])
	}
	modalities := []recipecontract.Modality{
		recipecontract.ModalityText, recipecontract.ModalityImage, recipecontract.ModalityAudio,
		recipecontract.ModalityVideo, recipecontract.ModalityTimeSeries, recipecontract.ModalityTable,
	}
	var every []string
	for _, input := range modalities {
		for _, output := range modalities {
			if pair := string(input) + "->" + string(output); pair != "text->image" && pair != "text->video" {
				every = append(every, pair)
			}
		}
	}
	if !slices.Equal(without, every) {
		t.Fatalf("pairs without an objective = %q, want every other pair %q", without, every)
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
	testcheck.NoError(t, run(clipObjectiveArgs(repository, "clip-corpus", "record", "train=3,heldout=1"), io.Discard))
	store := overgodbtest.OpenReadOnly(t, repository)
	objectiveID, found, err := store.ResolveAlias(ctx, "objective.registered.text-video")
	if err != nil || !found {
		t.Fatalf("objective alias = (%t, %v)", found, err)
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, objectiveID)
	testcheck.NoError(t, err)
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
	store, err := overgodb.Create(repository)
	testcheck.NoError(t, err)
	corpus := t.TempDir()
	for index := range clipCorpusFiles - 1 {
		// Distinct bytes per clip: identical files deduplicate to one record.
		clip := make([]byte, 2048)
		clip[0] = byte(index + 1)
		testcheck.NoError(t, os.WriteFile(filepath.Join(corpus, fmt.Sprintf("clip-%02d.mp4", index)), clip, 0o600))
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(corpus, "captions.json"), []byte(`[{"vid":"clip-00","caption":"c"}]`), 0o600))
	if _, err := dataset.RegisterDirectoryDataset(t.Context(), store, "clip-corpus", corpus); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
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

// TestObjectiveSplitResolvesThroughTheMaterializer holds a published split to
// the identities training reads records by: materializing the objective's
// dataset under its training membership selects exactly that membership's
// records. Record names of the split's own invention resolved none.
func TestObjectiveSplitResolvesThroughTheMaterializer(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	repository := registerClipCorpus(t)
	testcheck.NoError(t, run(clipObjectiveArgs(repository, "clip-corpus", "record", "train=3,heldout=1"), io.Discard))
	store := overgodbtest.OpenReadOnly(t, repository)
	objectiveID, found, err := store.ResolveAlias(ctx, "objective.registered.text-video")
	if err != nil || !found {
		t.Fatalf("objective alias = (%t, %v)", found, err)
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, objectiveID)
	testcheck.NoError(t, err)
	training, found, err := dataset.LoadMembership(ctx, store, objective.Split)
	if err != nil || !found {
		t.Fatalf("training membership = (%t, %v)", found, err)
	}
	processor := objective.Processors[0]
	materialized, err := trainingdata.Materialize(ctx, store, trainingdata.Authority{
		Dataset: objective.Dataset, Split: objective.Split, Processors: []artifact.ID{processor}, Signature: objective.Signature,
	}, []trainingdata.ProcessorBinding{{
		Artifact: processor, Modalities: slices.Concat(objective.Signature.Inputs, objective.Signature.Outputs),
		Process: func(_ context.Context, record trainingdata.RawRecord) (trainingdata.Example, error) {
			return trainingdata.Example{ID: record.ID, Group: record.Group}, nil
		},
	}})
	if err != nil {
		t.Fatalf("materializing the objective's training membership: %v", err)
	}
	defer materialized.Close()
	if records := materialized.Records(); records != len(training.Records) {
		t.Fatalf("materialized %d records, want the training membership's %d", records, len(training.Records))
	}
}

func mustChildren(t *testing.T, store *overgodb.Store, id artifact.ID) []artifact.ID {
	t.Helper()
	lineage, err := store.Children(t.Context(), id)
	testcheck.NoError(t, err)
	var children []artifact.ID
	for _, edge := range lineage {
		children = append(children, edge.Child)
	}
	return children
}
