//go:build windows

package adaptiveparity_test

import (
	"context"
	"fmt"
	"os/exec"
	"sort"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/controllertrain"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/workflowrecipe"
)

func TestControllerPromotionContract(t *testing.T) {
	cudatest.Require(t)
	commit := controllerSourceCommit(t)
	verifyControllerSources(t, commit)
	corpus, err := controllertrain.Compile(controllertrain.Spec{
		Train: controllerRecords(commit, false), Holdout: controllerRecords(commit, true),
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := make([]controllertrain.SeedRun, 3)
	for index, seed := range []int64{17, 29, 43} {
		runs[index], err = controllertrain.TrainSeed(corpus, seed, 800)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("seed %d: loss %.6f -> %.6f; action %.3f -> %.3f; modality %.3f -> %.3f",
			seed, runs[index].Evidence.Initial.Loss, runs[index].Evidence.Final.Loss,
			runs[index].Evidence.Initial.ActionAccuracy, runs[index].Evidence.Final.ActionAccuracy,
			runs[index].Evidence.Initial.ModalityAccuracy, runs[index].Evidence.Final.ModalityAccuracy)
	}
	sort.Slice(runs, func(i, j int) bool {
		left, right := runs[i].Evidence.Final, runs[j].Evidence.Final
		if left.ActionAccuracy != right.ActionAccuracy {
			return left.ActionAccuracy > right.ActionAccuracy
		}
		if left.ModalityAccuracy != right.ModalityAccuracy {
			return left.ModalityAccuracy > right.ModalityAccuracy
		}
		return left.Loss < right.Loss
	})
	pretrained := func(name string) artifact.ID {
		id, identifyErr := artifact.IdentifyBytes(artifact.KindModel, []byte(name))
		if identifyErr != nil {
			t.Fatal(identifyErr)
		}
		return id
	}
	incumbentEvaluation := runs[0].InitialEvaluation.ID
	incumbentMetrics := runs[0].Evidence.Initial
	decision, err := controllertrain.Decide(controllertrain.DecisionSpec{
		Dataset: corpus.Dataset(), Split: corpus.Split(),
		CurrentChampion: runs[0].InitialModel, Candidate: runs[0].Evidence.Model,
		Seeds: controllertrain.SeedEvidenceFrom(runs),
		Challengers: []controllertrain.Challenger{
			{Name: "current-champion", Model: runs[0].InitialModel, Evaluation: &incumbentEvaluation, Eligible: true, Metrics: &incumbentMetrics},
			{Name: "fractale-350m", Pretrained: true, Model: pretrained("fractale-350m"), Refusal: "controller-scorer-unavailable"},
			{Name: "carbon", Pretrained: true, Model: pretrained("carbon"), Refusal: "controller-scorer-unavailable"},
			{Name: "qwen2.5", Pretrained: true, Model: pretrained("qwen2.5"), Refusal: "controller-scorer-unavailable"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.State != controllertrain.DecisionPromote || decision.RollbackTarget() != runs[0].InitialModel {
		t.Fatalf("controller promotion = %s; rollback = %s", decision.State, decision.RollbackTarget())
	}
	publishControllerEvidence(t, corpus, runs, decision)
}

func controllerRecords(commit string, holdout bool) []controllertrain.Record {
	actions := []struct {
		prompt string
		action controllertrain.Action
	}{
		{"component text generate", controllertrain.Action{Scope: controllertrain.ScopeComponent, Task: recipe.TaskGeneration, Modality: controllertrain.ModalityText, Module: workflowrecipe.ModuleGenerate}},
		{"component image project", controllertrain.Action{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityImage, Module: workflowrecipe.ModuleProjectImage}},
		{"component audio project", controllertrain.Action{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityAudio, Module: workflowrecipe.ModuleProjectAudio}},
		{"component video project", controllertrain.Action{Scope: controllertrain.ScopeComponent, Task: recipe.TaskProjection, Modality: controllertrain.ModalityVideo, Module: workflowrecipe.ModuleProjectVideo}},
		{"workflow text train", controllertrain.Action{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityText, Module: workflowrecipe.ModuleBatchDataset}},
		{"workflow image train", controllertrain.Action{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityImage, Module: workflowrecipe.ModuleBatchDataset}},
		{"workflow audio train", controllertrain.Action{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityAudio, Module: workflowrecipe.ModuleBatchDataset}},
		{"workflow video train", controllertrain.Action{Scope: controllertrain.ScopeWorkflow, Task: recipe.TaskTraining, Modality: controllertrain.ModalityVideo, Module: workflowrecipe.ModuleBatchDataset}},
	}
	records := make([]controllertrain.Record, 0, len(actions)*10)
	for actionIndex, item := range actions {
		count := 10
		if holdout {
			count = 1
		}
		for variant := range count {
			caseNumber := variant
			split := "train"
			if holdout {
				caseNumber, split = 90+actionIndex, "holdout"
			}
			records = append(records, controllertrain.Record{
				ID:     fmt.Sprintf("%s-%02d-%02d", split, actionIndex, variant),
				Group:  fmt.Sprintf("%s-action-%02d", split, actionIndex),
				Source: controllertrain.GitSource{Commit: commit, Path: "internal/workflowrecipe/catalog.go"},
				Prompt: fmt.Sprintf("%s case %02d", item.prompt, caseNumber), Action: item.action,
			})
		}
	}
	return records
}

func controllerSourceCommit(t *testing.T) string {
	t.Helper()
	for _, revision := range []string{"MERGE_HEAD", "HEAD"} {
		command := exec.Command("git", "rev-parse", "--verify", revision)
		command.Dir = "../.."
		output, err := command.Output()
		if err == nil {
			return strings.TrimSpace(string(output))
		}
	}
	t.Fatal("Git source commit is unavailable")
	return ""
}

func verifyControllerSources(t *testing.T, commit string) {
	t.Helper()
	command := exec.Command("git", "show", commit+":internal/workflowrecipe/catalog.go")
	command.Dir = "../.."
	output, err := command.Output()
	if err != nil {
		t.Fatal(err)
	}
	for _, module := range []recipe.ModuleID{
		workflowrecipe.ModuleGenerate, workflowrecipe.ModuleProjectImage,
		workflowrecipe.ModuleProjectAudio, workflowrecipe.ModuleProjectVideo,
		workflowrecipe.ModuleBatchDataset,
	} {
		if !strings.Contains(string(output), string(module)) {
			t.Fatalf("Git source lacks controller action module %q", module)
		}
	}
}

func publishControllerEvidence(t *testing.T, corpus controllertrain.Corpus, runs []controllertrain.SeedRun, decision controllertrain.Decision) {
	t.Helper()
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := corpus.PublicationBatch("controller/corpus")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	descriptors := map[artifact.ID]artifact.Descriptor{}
	for _, run := range runs {
		for _, id := range []artifact.ID{run.Recipe, run.ScratchDataset, run.ScratchSplit, run.InitialModel, run.Evidence.Model} {
			descriptors[id] = artifact.Descriptor{ID: id}
		}
	}
	for _, challenger := range decision.Challengers {
		descriptors[challenger.Model] = artifact.Descriptor{ID: challenger.Model}
	}
	facts := make([]artifact.Descriptor, 0, len(descriptors))
	for _, descriptor := range descriptors {
		facts = append(facts, descriptor)
	}
	sort.Slice(facts, func(i, j int) bool { return facts[i].ID.String() < facts[j].ID.String() })
	if _, err := store.Commit(ctx, artifact.Batch{Key: "controller/facts", Artifacts: facts}); err != nil {
		t.Fatal(err)
	}
	for index, run := range runs {
		for _, value := range []struct {
			suffix    string
			runBatch  func(string) (artifact.Batch, error)
			evalBatch func(string) (artifact.Batch, error)
		}{
			{"initial", run.InitialRun.Batch, run.InitialEvaluation.Batch},
			{"final", run.Run.Batch, run.Evaluation.Batch},
		} {
			runBatch, err := value.runBatch(fmt.Sprintf("controller/seed-%d/%s/run", index, value.suffix))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(ctx, runBatch); err != nil {
				t.Fatal(err)
			}
			evaluationBatch, err := value.evalBatch(fmt.Sprintf("controller/seed-%d/%s/evaluation", index, value.suffix))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(ctx, evaluationBatch); err != nil {
				t.Fatal(err)
			}
		}
	}
	decisionBatch, err := decision.Batch("controller/promotion")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, decisionBatch); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := store.Content(ctx, decision.ID); err != nil || !ok {
		t.Fatalf("promotion evidence stored = %v, %v", ok, err)
	}
}
