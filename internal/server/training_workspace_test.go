package server

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

func TestTrainingWorkspacePublishesEvaluationRequiredDecision(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	roots := dataroot.Roots{Models: filepath.Join(root, "models"), Datasets: filepath.Join(root, "datasets"), Checkpoints: filepath.Join(root, "checkpoints")}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 9,
	})
	referenceWeights, referenceShapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 10,
	})
	policyPath := filepath.Join(roots.Models, "policy")
	referencePath := filepath.Join(roots.Models, "reference")
	policy := writeWorkspaceModel(t, policyPath, weights, shapes)
	reference := writeWorkspaceModel(t, referencePath, referenceWeights, referenceShapes)
	datasetData := []byte("{\"id\":\"pair\",\"prompt\":\"ab\",\"chosen\":\"c\",\"rejected\":\"d\"}\n")
	if err := os.MkdirAll(roots.Datasets, 0o755); err != nil {
		t.Fatal(err)
	}
	datasetPath := filepath.Join(roots.Datasets, "preference.jsonl")
	if err := os.WriteFile(datasetPath, datasetData, 0o600); err != nil {
		t.Fatal(err)
	}
	dataset, err := artifact.IdentifyBytes(artifact.KindDataset, datasetData)
	if err != nil {
		t.Fatal(err)
	}
	split, err := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), datasetData...))
	if err != nil {
		t.Fatal(err)
	}
	processor := testutil.ArtifactID(t, artifact.KindProfile, "overgo/training/preference-token-pair/v1")
	evaluation := testutil.ArtifactID(t, artifact.KindProfile, "dpo evaluation")
	policies := trainingprogram.PolicySpec{
		Precision:  testutil.ArtifactID(t, artifact.KindProfile, "dpo precision"),
		Placement:  testutil.ArtifactID(t, artifact.KindProfile, "dpo placement"),
		Memory:     testutil.ArtifactID(t, artifact.KindProfile, "dpo memory"),
		Checkpoint: testutil.ArtifactID(t, artifact.KindProfile, "dpo checkpoint"),
		Evaluation: evaluation,
		Promotion:  testutil.ArtifactID(t, artifact.KindProfile, "dpo promotion"),
	}
	optimizerPolicy := trainingprogram.BuiltinOptimizerPolicy()
	policies.Optimizer = optimizerPolicy.ID
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "dpo objective evidence")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "dpo", Kind: trainingprogram.ObjectiveDPO,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: dataset, Split: split, Processors: []artifact.ID{processor},
		Loss: testutil.ArtifactID(t, artifact.KindProfile, "dpo loss"), Evaluation: evaluation,
		Metric: trainingprogram.MetricTokenAccuracy, Evidence: []artifact.ID{evidence},
		Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	policies.Objective = objective.ID
	objectiveContent, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	optimizerContent, err := optimizerPolicy.Content()
	if err != nil {
		t.Fatal(err)
	}
	authorities := []artifact.ID{
		split, processor, objective.Loss, evidence, policies.Precision, policies.Placement,
		policies.Memory, policies.Checkpoint, policies.Evaluation, policies.Promotion,
	}
	descriptors := make([]artifact.Descriptor, len(authorities))
	for index, id := range authorities {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	const championAlias = "models/training-workspace/champion"
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/training-workspace/artifacts",
		Artifacts: append(descriptors, []artifact.Descriptor{
			{ID: policy, Size: 1}, {ID: reference, Size: 1}, {ID: dataset, Size: uint64(len(datasetData))},
		}...),
		Contents: []artifact.Content{objectiveContent, optimizerContent},
		Locations: []artifact.LocationEvent{
			location(policy, artifact.LocationDirectory, policyPath),
			location(reference, artifact.LocationDirectory, referencePath),
			location(dataset, artifact.LocationFile, datasetPath),
		},
		Aliases: []artifact.AliasBinding{{Name: championAlias, Target: policy}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := dpoFixtureDefinition(append([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: policy},
		{Role: recipe.DependencyModel, Slot: 1, Artifact: reference},
	}, policyDependencies(policies)...))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/training-workspace/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/training-workspace/verification", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "native DPO fixture"); err != nil {
		t.Fatal(err)
	}
	workspace, err := NewTrainingWorkspace(ctx, store, roots, policy)
	if err != nil {
		t.Fatal(err)
	}
	capabilities, err := workspace.WorkflowCapabilities(ctx, WorkflowTraining)
	if err != nil || len(capabilities) != 1 || capabilities[0].Recipe != definition.ID ||
		capabilities[0].Controls[0].Type != WorkflowControlDataset {
		t.Fatalf("capabilities=%+v err=%v", capabilities, err)
	}
	expectedObservations := []struct{}{{}}
	input, err := json.Marshal(map[string]any{
		"dataset": dataset, "output": "trained", "steps": len(expectedObservations),
		"objective_scale": 0.1,
	})
	if err != nil {
		t.Fatal(err)
	}
	completion, err := workspace.ExecuteWorkflow(ctx, WorkflowTraining, recipe.TaskTraining, definition.ID, input, testReporter{})
	if err != nil || completion.Run.Kind() != artifact.KindRun || len(completion.Outputs) != 3 ||
		completion.Outputs[0].Kind() != artifact.KindCheckpoint || completion.Outputs[1].Kind() != artifact.KindEvidence ||
		completion.Outputs[2].Kind() != artifact.KindEvidence {
		t.Fatalf("completion=%+v err=%v", completion, err)
	}
	if checkpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(roots.Checkpoints, "trained")); err != nil || checkpoint.ID() != completion.Outputs[0] {
		t.Fatalf("checkpoint=%s err=%v", checkpoint.ID(), err)
	}
	traceContent, ok, err := artifact.ReadContent(ctx, store, completion.Outputs[1])
	if err != nil || !ok {
		t.Fatalf("training trace content exists=%v err=%v", ok, err)
	}
	var trace runrecord.TrainingTrace
	err = json.Unmarshal(traceContent.Data, &trace)
	if err != nil || trace.Run != completion.Run || trace.Recipe != definition.ID ||
		trace.Dataset != dataset || trace.Policy != policy || trace.Reference != reference ||
		trace.Objective != trainingprogram.ObjectiveDPO || len(trace.DPO) != len(expectedObservations) {
		t.Fatalf("trace=%+v err=%v", trace, err)
	}
	decisionContent, ok, err := artifact.ReadContent(ctx, store, completion.Outputs[2])
	if err != nil || !ok {
		t.Fatalf("training decision content exists=%v err=%v", ok, err)
	}
	var decision runrecord.TrainingDecision
	err = json.Unmarshal(decisionContent.Data, &decision)
	if err != nil || decision.State != runrecord.TrainingEvaluationRequired ||
		decision.Run != completion.Run || decision.Recipe != definition.ID || decision.Parent != policy ||
		decision.Checkpoint != completion.Outputs[0] || decision.Trace != completion.Outputs[1] || decision.Rollback != policy {
		t.Fatalf("decision=%+v err=%v", decision, err)
	}
	champion, ok, err := store.ResolveAlias(ctx, championAlias)
	if err != nil || !ok || champion != policy {
		t.Fatalf("champion after training=(%s,%v,%v), want unchanged parent %s", champion, ok, err, policy)
	}
}

func TestTrainingWorkspaceAdmissionMatrix(t *testing.T) {
	cases := []struct {
		name       string
		objective  trainingprogram.ObjectiveKind
		registered bool
		admitted   bool
	}{
		{name: "dpo", objective: trainingprogram.ObjectiveDPO, registered: true, admitted: true},
		{name: "grpo", objective: trainingprogram.ObjectiveGRPO, registered: true, admitted: true},
		{name: "token", objective: trainingprogram.ObjectiveTokenPrediction, registered: true},
		{name: "unregistered-model", objective: trainingprogram.ObjectiveDPO},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			ctx := context.Background()
			root := t.TempDir()
			store, err := overgodb.Open(filepath.Join(root, "store"))
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			policy := testutil.ArtifactID(t, artifact.KindModel, test.name+" policy")
			reference := testutil.ArtifactID(t, artifact.KindModel, test.name+" reference")
			dependencies := []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: policy}}
			artifacts := []artifact.Descriptor{{ID: policy}}
			if test.objective == trainingprogram.ObjectiveDPO {
				dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyModel, Slot: 1, Artifact: reference})
				artifacts = append(artifacts, artifact.Descriptor{ID: reference})
			}
			definition, err := trainingWorkspaceAdmissionDefinition(test.objective, dependencies)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/training-admission/artifacts", Artifacts: artifacts}); err != nil {
				t.Fatal(err)
			}
			if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/training-admission/candidate", definition); err != nil {
				t.Fatal(err)
			}
			verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/training-admission/verification", definition.ID)
			if err != nil {
				t.Fatal(err)
			}
			if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "training admission fixture"); err != nil {
				t.Fatal(err)
			}
			model := testutil.ArtifactID(t, artifact.KindModel, test.name+" absent model")
			if test.registered {
				model = policy
			}
			workspace, err := NewTrainingWorkspace(ctx, store, dataroot.Roots{Checkpoints: filepath.Join(root, "checkpoints")}, model)
			if (err == nil) != test.admitted {
				t.Fatalf("workspace admitted=%v err=%v, want admitted=%v", err == nil, err, test.admitted)
			}
			if !test.admitted {
				return
			}
			capabilities, err := workspace.WorkflowCapabilities(ctx, WorkflowTraining)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, capability := range capabilities {
				found = found || capability.Task == recipe.TaskTraining && capability.Recipe == definition.ID
			}
			if !found {
				t.Fatalf("active training recipe %s absent from GUI capabilities", definition.ID)
			}
		})
	}
}

func writeWorkspaceModel(t *testing.T, directory string, weights map[string][]float32, shapes map[string][]int) artifact.ID {
	t.Helper()
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, trainingprogram.CheckpointWeights)
	if err := safetensors.Save(path, weights, shapes, nil); err != nil {
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
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	id, _, identifyErr := artifact.Identify(artifact.KindModel, file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if identifyErr != nil {
		t.Fatal(identifyErr)
	}
	return id
}

func location(id artifact.ID, kind artifact.LocationKind, path string) artifact.LocationEvent {
	return artifact.LocationEvent{Location: artifact.Location{Artifact: id, Kind: kind, Value: path}, Action: artifact.LocationAdd}
}

func dpoFixtureDefinition(dependencies []recipe.Dependency) (recipe.Definition, error) {
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

func trainingWorkspaceAdmissionDefinition(objective trainingprogram.ObjectiveKind, dependencies []recipe.Dependency) (recipe.Definition, error) {
	switch objective {
	case trainingprogram.ObjectiveDPO:
		return dpoFixtureDefinition(dependencies)
	case trainingprogram.ObjectiveGRPO:
		nodes := []recipe.Node{
			{ID: "batch", Module: workflowrecipe.ModuleBatchRollout, Placement: recipe.PlacementHost},
			{ID: "policy", Module: workflowrecipe.ModuleScorePolicy, Placement: recipe.PlacementHost},
			{ID: "objective", Module: workflowrecipe.ModuleGRPOObjective, Placement: recipe.PlacementHost},
			{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
			{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
		}
		return linearTrainingDefinition(dependencies, nodes, []recipe.Edge{
			trainingEdge("batch", "batch", "policy", "batch"),
			trainingEdge("policy", "scores", "objective", "policy"),
			trainingEdge("objective", "loss", "backward", "loss"),
			trainingEdge("backward", "gradients", "optimize", "gradients"),
		})
	case trainingprogram.ObjectiveTokenPrediction:
		nodes := []recipe.Node{
			{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost},
			{ID: "forward", Module: workflowrecipe.ModuleTrainingForward, Placement: recipe.PlacementHost},
			{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
			{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
		}
		return linearTrainingDefinition(dependencies, nodes, []recipe.Edge{
			trainingEdge("batch", "batch", "forward", "batch"),
			trainingEdge("forward", "loss", "backward", "loss"),
			trainingEdge("backward", "gradients", "optimize", "gradients"),
		})
	default:
		return recipe.Definition{}, os.ErrInvalid
	}
}

func linearTrainingDefinition(dependencies []recipe.Dependency, nodes []recipe.Node, edges []recipe.Edge) (recipe.Definition, error) {
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes, edges, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
}

func trainingEdge(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
	return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
}

func policyDependencies(spec trainingprogram.PolicySpec) []recipe.Dependency {
	return []recipe.Dependency{
		{Role: recipe.DependencyObjective, Artifact: spec.Objective},
		{Role: recipe.DependencyPrecision, Artifact: spec.Precision},
		{Role: recipe.DependencyPlacement, Artifact: spec.Placement},
		{Role: recipe.DependencyMemory, Artifact: spec.Memory},
		{Role: recipe.DependencyOptimizer, Artifact: spec.Optimizer},
		{Role: recipe.DependencyCheckpointPolicy, Artifact: spec.Checkpoint},
		{Role: recipe.DependencyEvaluation, Artifact: spec.Evaluation},
		{Role: recipe.DependencyPromotion, Artifact: spec.Promotion},
	}
}

type testReporter struct{ id artifact.ID }

func (reporter testReporter) OperationID() artifact.ID { return reporter.id }
func (testReporter) Attempt(artifact.ID)               {}
func (testReporter) Progress(uint64, *uint64)          {}
func (testReporter) Metric(operation.Metric)           {}
func (testReporter) Publishing()                       {}
