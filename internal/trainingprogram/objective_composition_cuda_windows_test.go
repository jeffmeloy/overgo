//go:build windows

package trainingprogram_test

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/controlleraction"
	"overgo/internal/controllertrain"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/optimizer"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/repodb"
	"overgo/internal/scratchmodel"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

func TestComposedObjectiveProducesBetterDescendant(t *testing.T) {
	cudatest.Require(t)
	baseRecords, addedRecords, holdout := objectiveRecords()
	base := objectiveCorpus(t, baseRecords, holdout)
	added := objectiveCorpus(t, addedRecords, holdout)
	aggregate := objectiveCorpus(t, append(append([]controllertrain.Record{}, baseRecords...), addedRecords...), holdout)

	ctx := context.Background()
	store, err := repodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for index, corpus := range []controllertrain.Corpus{base, added, aggregate} {
		batch, err := corpus.PublicationBatch(fmt.Sprintf("objective-corpus-%d", index))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}

	processor := testutil.ArtifactID(t, artifact.KindProfile, "controller processor")
	loss := testutil.ArtifactID(t, artifact.KindProfile, "token loss")
	evaluator := testutil.ArtifactID(t, artifact.KindProfile, "controller evaluator")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "objective evidence")
	contract := trainingprogram.ObjectiveSpec{
		Name: "controller aggregate", Kind: trainingprogram.ObjectiveTokenPrediction,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Processors: []artifact.ID{processor}, Loss: loss, Evaluation: evaluator,
		Metric: trainingprogram.MetricTokenAccuracy, Evidence: []artifact.ID{evidence},
		Authority: trainingprogram.ObjectiveAdaptive,
	}
	objectives := make([]trainingprogram.ObjectiveDocument, 2)
	for index, corpus := range []controllertrain.Corpus{base, added} {
		spec := contract
		spec.Name = fmt.Sprintf("controller component %d", index)
		spec.Dataset, spec.Split = corpus.Dataset(), corpus.Split()
		objectives[index], err = trainingprogram.NewObjective(spec)
		if err != nil {
			t.Fatal(err)
		}
		content, err := objectives[index].Content()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, artifact.Batch{Key: fmt.Sprintf("objective-%d", index), Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "objective-facts", Artifacts: []artifact.Descriptor{
		{ID: processor}, {ID: loss}, {ID: evaluator}, {ID: evidence},
	}}); err != nil {
		t.Fatal(err)
	}
	contract.Dataset, contract.Split = aggregate.Dataset(), aggregate.Split()
	composition, err := trainingprogram.NewObjectiveComposition(trainingprogram.ObjectiveCompositionSpec{
		ObjectiveSpec: contract, Components: []artifact.ID{objectives[0].ID, objectives[1].ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	action := controlleraction.Action{
		Version: controlleraction.ActionVersion, Kind: controlleraction.KindObjectiveComposition,
		Objective: &controlleraction.ObjectiveCompositionAction{Spec: composition.ObjectiveCompositionSpec},
	}
	batch, err := controlleraction.CompileTransaction(ctx, store, action)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	compileObjectiveRun(t, ctx, store, composition, contract)

	const trainingSteps = 400
	profile := scratchmodel.AdaptiveDerivationProfile()
	incumbent, err := controllertrain.TrainSeed(base, profile, 17, trainingSteps)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := controllertrain.TrainSeed(aggregate, profile, 17, trainingSteps)
	if err != nil {
		t.Fatal(err)
	}
	if candidate.Evidence.Final.Loss >= incumbent.Evidence.Final.Loss ||
		candidate.Evidence.Final.ActionAccuracy < incumbent.Evidence.Final.ActionAccuracy ||
		candidate.Evidence.Final.ModalityAccuracy < incumbent.Evidence.Final.ModalityAccuracy {
		t.Fatalf("composed descendant did not improve: incumbent=%+v candidate=%+v", incumbent.Evidence.Final, candidate.Evidence.Final)
	}
	t.Logf("composed objective action %.3f -> %.3f; modality %.3f -> %.3f", incumbent.Evidence.Final.ActionAccuracy,
		candidate.Evidence.Final.ActionAccuracy, incumbent.Evidence.Final.ModalityAccuracy, candidate.Evidence.Final.ModalityAccuracy)
}

func compileObjectiveRun(t *testing.T, ctx context.Context, store artifact.Reader, composition trainingprogram.ObjectiveComposition, objective trainingprogram.ObjectiveSpec) {
	t.Helper()
	plan, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveTokenPrediction, nil, plan)
	if err != nil {
		t.Fatal(err)
	}
	profile := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindProfile, name) }
	_, err = trainingprogram.CompileTrainingRunPlanFromRepository(ctx, store, trainingprogram.RunSpec{
		Recipe:  testutil.ArtifactID(t, artifact.KindRecipe, "composed training"),
		Initial: trainingprogram.InitialStateSpec{Model: testutil.ArtifactID(t, artifact.KindModel, "incumbent")},
		Dataset: objective.Dataset, Split: objective.Split, Signature: objective.Signature,
		Processors: objective.Processors, Projectors: objective.Projectors, Codecs: objective.Codecs,
		Policies: trainingprogram.PolicySpec{
			Objective: composition.ID, Precision: profile("precision"), Placement: profile("placement"),
			Memory: profile("memory"), Checkpoint: profile("checkpoint"), Evaluation: objective.Evaluation,
			Promotion: profile("promotion"),
		},
		Program: program,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func objectiveCorpus(t *testing.T, train, holdout []controllertrain.Record) controllertrain.Corpus {
	t.Helper()
	corpus, err := controllertrain.Compile(controllertrain.Spec{Train: train, Holdout: holdout})
	if err != nil {
		t.Fatal(err)
	}
	return corpus
}

func objectiveRecords() (base, added, holdout []controllertrain.Record) {
	actions := []controllertrain.Action{
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskGeneration, Modality: controllertrain.ModalityText, Module: workflowrecipe.ModuleGenerate},
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityImage, Module: workflowrecipe.ModuleProjectImage},
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityAudio, Module: workflowrecipe.ModuleProjectAudio},
		{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityVideo, Module: workflowrecipe.ModuleProjectVideo},
		{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityText, Module: workflowrecipe.ModuleBatchDataset},
		{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityImage, Module: workflowrecipe.ModuleBatchDataset},
		{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityAudio, Module: workflowrecipe.ModuleBatchDataset},
		{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityVideo, Module: workflowrecipe.ModuleBatchDataset},
	}
	prompts := []string{
		"component text generate", "component image project", "component audio project", "component video project",
		"workflow text train", "workflow image train", "workflow audio train", "workflow video train",
	}
	const sourceCommit = "0123456789abcdef0123456789abcdef01234567"
	for actionIndex, action := range actions {
		makeRecord := func(part, prompt string, variant int) controllertrain.Record {
			return controllertrain.Record{
				ID: fmt.Sprintf("%s-%02d-%02d", part, actionIndex, variant), Group: fmt.Sprintf("%s-%02d-%02d", part, actionIndex, variant),
				Source: controllertrain.GitSource{Commit: sourceCommit, Path: "internal/workflowrecipe/catalog.go"},
				Prompt: prompt, Action: action,
			}
		}
		base = append(base, makeRecord("base", "abcdefghijklmnopqrstuvwxyz 0123456789", 0))
		for variant := 1; variant < 10; variant++ {
			added = append(added, makeRecord("added", fmt.Sprintf("%s 0123456789 case %02d", prompts[actionIndex], variant), variant))
		}
		holdout = append(holdout, makeRecord("holdout", fmt.Sprintf("%s 0123456789 case %02d", prompts[actionIndex], 90+actionIndex), 90+actionIndex))
	}
	return base, added, holdout
}
