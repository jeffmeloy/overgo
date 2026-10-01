package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestEveryBlockingProofHasACommand holds every proof promotion waits on to
// a command that re-acquires it into the selected store. Each training recipe
// declares an objective over the data its route grounds on, and the record
// reader names the route from that objective alone: the dit route trains a
// video stack from its directory and a raw checkpoint against that stack, and
// the dense workflow trains the recipe it is named. A recipe over data no
// route grounds binds nothing, and a projection proof says which producer it
// lacks.
func TestEveryBlockingProofHasACommand(t *testing.T) {
	t.Parallel()
	ctx, root, models := t.Context(), testutil.RepoRoot(t), t.TempDir()
	const selected = "C:/stores/selected"
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publish := func(content artifact.Content, err error) artifact.ID {
		t.Helper()
		if err == nil {
			_, err = artifact.Publish(ctx, store, artifact.Batch{Key: content.Descriptor.ID.String(), Contents: []artifact.Content{content}})
		}
		if err != nil {
			t.Fatal(err)
		}
		return content.Descriptor.ID
	}
	training := func(name, weights, grounding string) ModelValidation {
		data := []byte("data no route grounds")
		if grounding != "" {
			if data, err = os.ReadFile(filepath.Join(root, filepath.FromSlash(grounding))); err != nil {
				t.Fatal(err)
			}
		}
		profile := testutil.ArtifactID(t, artifact.KindProfile, name)
		objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
			Name: name, Kind: trainingprogram.ObjectiveTokenPrediction,
			Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
			Dataset:   testutil.ArtifactBytesID(t, artifact.KindDataset, data), Split: testutil.ArtifactID(t, artifact.KindDatasetShard, name),
			Processors: []artifact.ID{profile}, Loss: profile, Evaluation: profile, Metric: trainingprogram.MetricTokenAccuracy,
			Evidence: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, name)}, Authority: trainingprogram.ObjectiveApproved,
		})
		if err != nil {
			t.Fatal(err)
		}
		model := testutil.ArtifactID(t, artifact.KindModel, name)
		definition, err := recipe.NewDefinitionWithDependencies(recipe.TaskTraining,
			[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: model}, {Role: recipe.DependencyObjective, Artifact: publish(objective.Content())}},
			[]recipe.Node{{ID: "optimize", Module: "training.optimize", Placement: recipe.PlacementHost}}, nil, nil,
			[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}})
		if err != nil {
			t.Fatal(err)
		}
		return ModelValidation{
			Model: model, ModelName: name, Location: filepath.Join(models, name, weights),
			Kind: recipe.TaskTraining, Validation: trainingProof, Surface: "training", Recipe: publish(definition.ArtifactContent()),
		}
	}
	stack, edit := training("Wan", ditWeights, ditGrounding), training("LiveEdit", "edit.pt", ditGrounding)
	mot := training("RxBrain", "model-00001-of-00003.safetensors", "docs/verification/rxbrain-mot-training-stimulus.txt")
	hybrid := training("Qwen35", "qwen35.gguf", "fixtures/qwen35_4b_serving_golden.json")
	dense, ungrounded := training("Qwen25", "model.safetensors", denseGrounding), training("Unrouted", "model.safetensors", "")
	cells := []ModelValidation{stack, edit, mot, hybrid, dense, ungrounded}
	affected := map[SurfaceID]AffectRecord{"training": {Affected: true, Reason: "fixture"}}
	for _, task := range []recipe.Task{
		recipe.TaskProjection, recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskSpeech, recipe.TaskTranscription,
		recipe.TaskVQA, recipe.TaskSeq2Seq, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskImageEmbedding,
	} {
		for _, validation := range kindValidation(task) {
			affected[validation.Surface] = AffectRecord{Affected: true, Reason: "fixture"}
			cells = append(cells, ModelValidation{
				Model: testutil.ArtifactID(t, artifact.KindModel, string(task)), ModelName: string(task), Location: "C:/models/" + string(task),
				Kind: task, Validation: validation.Validation, Surface: validation.Surface, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, string(task)),
			})
		}
	}
	if err := groundTraining(ctx, root, store, cells); err != nil {
		t.Fatal(err)
	}
	plan := SelectValidation(Inputs{AffectedSurfaces: affected, Registered: cells, Current: func(ModelValidation) error { return nil }})
	if len(plan.Run) != len(cells) {
		t.Fatalf("promotion waits on %d of %d proofs: background=%+v", len(plan.Run), len(cells), plan.Background)
	}

	refusals := map[string]string{string(recipe.TaskProjection): "re-verifies the active recipe", ungrounded.ModelName: "no cmd/train route grounds"}
	commands, environments := map[string][]string{}, map[string][]string{}
	for _, sel := range plan.Run {
		_, environments[sel.ModelName], _, _ = trainingRun(sel.ModelValidation, selected)
		name, args, err := reacquireCommand(sel.ModelValidation, selected)
		if refusal, refused := refusals[sel.ModelName]; refused {
			if err == nil || !strings.Contains(err.Error(), refusal) {
				t.Fatalf("%s %s bound (%v, %v), want the refusal that it %s", sel.ModelName, sel.Validation, args, err, refusal)
			}
			continue
		}
		if err != nil || name != "go" || !slices.Contains(args, selected) {
			t.Fatalf("%s %s bound (%q, %v, %v), want a command into the selected store", sel.ModelName, sel.Validation, name, args, err)
		}
		commands[sel.ModelName] = args
	}
	flag := func(model, name string) string {
		args := commands[model]
		if at := slices.Index(args, name); at >= 0 && at+1 < len(args) {
			return args[at+1]
		}
		return ""
	}
	for model, want := range map[string]map[string]string{
		stack.ModelName:  {"-route": "dit", "-model-dir": filepath.Dir(stack.Location), "-checkpoint": ""},
		edit.ModelName:   {"-route": "dit", "-model-dir": filepath.Dir(stack.Location), "-checkpoint": edit.Location},
		mot.ModelName:    {"-route": "mot", "-model": filepath.Dir(mot.Location)},
		hybrid.ModelName: {"-route": "hybrid-stack", "-model": hybrid.Location},
		dense.ModelName:  {"-route": "", "-recipe": dense.Recipe.String(), "-model": filepath.Dir(dense.Location), "-dataset": denseGrounding},
	} {
		for name, value := range want {
			if got := flag(model, name); got != value {
				t.Errorf("%s runs %s %q, want %q: %v", model, name, got, value, commands[model])
			}
		}
		if out := flag(model, "-out"); !strings.HasPrefix(out, os.TempDir()) {
			t.Errorf("%s publishes its checkpoint at %q, outside the scratch -run empties", model, out)
		}
	}
	if !slices.Contains(environments[hybrid.ModelName], "GOMEMLIMIT=68GiB") {
		t.Errorf("the hybrid stack runs without the memory limit its route was recorded under: %v", environments[hybrid.ModelName])
	}
}
