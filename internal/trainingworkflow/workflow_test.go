package trainingworkflow

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/repodb"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

func TestTrainingWorkflowRequiresStoredAuthority(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 6,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, []byte("ab"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "missing active recipe")
	_, err = Execute(context.Background(), Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "output"), Steps: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "no active training recipe") {
		t.Fatalf("absent authority error = %v", err)
	}
}

func TestNativeTrainingWorkflowDPOExactResume(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 7,
	})
	root := t.TempDir()
	policy := filepath.Join(root, "policy")
	reference := filepath.Join(root, "reference")
	writeModel(t, policy, weights, shapes)
	writeModel(t, reference, weights, shapes)
	dataset := filepath.Join(root, "preference.jsonl")
	if err := os.WriteFile(dataset, []byte("{\"id\":\"pair\",\"prompt\":\"ab\",\"chosen\":\"c\",\"rejected\":\"d\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := dpoAuthority(t, policy, reference, dataset)
	request := Request{
		Repository: store, Recipe: recipeID,
		ModelDirectory: policy, ReferenceDirectory: reference, DatasetPath: dataset,
		Steps: 2, LearningRate: 0, Momentum: 0.9, DPOScale: 0.1, Host: true,
	}
	request.OutputDirectory = filepath.Join(root, "uninterrupted")
	want, err := Execute(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.Steps = 1
	request.OutputDirectory = filepath.Join(root, "first")
	first, err := Execute(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.ModelDirectory = ""
	request.ResumeDirectory = request.OutputDirectory
	request.OutputDirectory = filepath.Join(root, "resumed")
	got, err := Execute(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	wantModel, err := densecausal.Load(filepath.Join(root, "uninterrupted"))
	if err != nil {
		t.Fatal(err)
	}
	gotModel, err := densecausal.Load(filepath.Join(root, "resumed"))
	if err != nil {
		t.Fatal(err)
	}
	wantCheckpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(root, "uninterrupted"))
	if err != nil {
		t.Fatal(err)
	}
	gotCheckpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(root, "resumed"))
	if err != nil {
		t.Fatal(err)
	}
	if want.Backend != "host" || first.StreamPosition+1 != got.StreamPosition ||
		!reflect.DeepEqual(wantModel.Weights, gotModel.Weights) ||
		!reflect.DeepEqual(wantCheckpoint.Optimizer, gotCheckpoint.Optimizer) {
		t.Fatalf("DPO resume differs: want=%+v first=%+v got=%+v", want, first, got)
	}
}

func writeModel(t *testing.T, directory string, weights map[string][]float32, shapes map[string][]int) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	tokenizer := `{"model":{"type":"BPE","vocab":{"a":1,"b":2,"c":3,"d":4},"merges":[]}}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(tokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
}

func dpoAuthority(t *testing.T, policyDirectory, referenceDirectory, datasetPath string) (*repodb.Store, artifact.ID) {
	t.Helper()
	ctx := context.Background()
	store, err := repodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	policy, err := identifyModel(policyDirectory)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := identifyModel(referenceDirectory)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	dataset, _ := artifact.IdentifyBytes(artifact.KindDataset, raw)
	split, _ := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), raw...))
	processor, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/preference-token-pair/v1"))
	profile := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindProfile, name) }
	evaluation := profile("dpo evaluation")
	policies := trainingprogram.PolicySpec{
		Precision: profile("dpo precision"), Placement: profile("dpo placement"), Memory: profile("dpo memory"),
		Checkpoint: profile("dpo checkpoint"), Evaluation: evaluation, Promotion: profile("dpo promotion"),
	}
	loss, evidence := profile("dpo loss"), testutil.ArtifactID(t, artifact.KindEvidence, "dpo evidence")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "dpo", Kind: trainingprogram.ObjectiveDPO,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: dataset, Split: split, Processors: []artifact.ID{processor}, Loss: loss,
		Evaluation: evaluation, Metric: trainingprogram.MetricTokenAccuracy,
		Evidence: []artifact.ID{evidence}, Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	policies.Objective = objective.ID
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	ids := []artifact.ID{
		policy, reference, dataset, split, processor, loss, evidence, policies.Precision,
		policies.Placement, policies.Memory, policies.Checkpoint, policies.Evaluation, policies.Promotion,
	}
	descriptors := make([]artifact.Descriptor, len(ids))
	for index, id := range ids {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "training/workflow/authority", Artifacts: descriptors, Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	definition, err := dpoDefinition(append([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: policy},
		{Role: recipe.DependencyModel, Slot: 1, Artifact: reference},
	}, policyDependencies(policies)...))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "training/workflow/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "training/workflow/verification", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceExperimental, "training workflow fixture"); err != nil {
		t.Fatal(err)
	}
	return store, definition.ID
}

func dpoDefinition(dependencies []recipe.Dependency) (recipe.Definition, error) {
	nodes := []recipe.Node{
		{ID: "batch", Module: workflowrecipe.ModuleBatchPreference, Placement: recipe.PlacementHost},
		{ID: "policy", Module: workflowrecipe.ModuleScorePolicy, Placement: recipe.PlacementHost},
		{ID: "reference", Module: workflowrecipe.ModuleScoreReference, Placement: recipe.PlacementHost, ModelSlot: 1},
		{ID: "objective", Module: workflowrecipe.ModuleDPOObjective, Placement: recipe.PlacementHost},
		{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
		{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
	}
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes,
		[]recipe.Edge{
			edge("batch", "batch", "policy", "batch"), edge("batch", "batch", "reference", "batch"),
			edge("policy", "scores", "objective", "policy"), edge("reference", "scores", "objective", "reference"),
			edge("objective", "loss", "backward", "loss"), edge("backward", "gradients", "optimize", "gradients"),
		}, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
}

func policyDependencies(spec trainingprogram.PolicySpec) []recipe.Dependency {
	return []recipe.Dependency{
		{Role: recipe.DependencyObjective, Artifact: spec.Objective},
		{Role: recipe.DependencyPrecision, Artifact: spec.Precision},
		{Role: recipe.DependencyPlacement, Artifact: spec.Placement},
		{Role: recipe.DependencyMemory, Artifact: spec.Memory},
		{Role: recipe.DependencyCheckpointPolicy, Artifact: spec.Checkpoint},
		{Role: recipe.DependencyEvaluation, Artifact: spec.Evaluation},
		{Role: recipe.DependencyPromotion, Artifact: spec.Promotion},
	}
}
