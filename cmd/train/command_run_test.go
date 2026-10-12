package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/densecausal"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
	"overgo/internal/workflowrecipe"
)

// commandChildEnvironment, set in a child of this test binary, has the child
// run as the training command would, in the way its value names, and run no
// test.
const commandChildEnvironment = "OVERGO_TEST_TRAIN_COMMAND"

const (
	// childStandIn runs the command's own main with a device trainer that
	// steps on no gradient, so a run off the host needs no graphics card. It
	// prints childStartedLine first, so what it prints is more than its error.
	childStandIn     = "stand-in"
	childStartedLine = "the stand-in training command started"
	// childAnnouncesStoreWait is childStandIn that also prints storeWaitLine
	// once the command is waiting for the store's writer lock.
	childAnnouncesStoreWait = "stand-in that announces its wait for the store"
	storeWaitLine           = "the stand-in training command is waiting for the store's writer lock"
	// childDevice runs the command's own main as it is built.
	childDevice = "device"
	// childFails prints childFailureLine and ends with childFailureStatus,
	// writing no error file, as a command that dies outside its own error
	// path does.
	childFails         = "fails"
	childFailureLine   = "the stand-in training command gave up"
	childFailureStatus = 3
	// childOfAnotherBuild ends well and hands back no steps, as a training
	// command built before it handed its steps back does.
	childOfAnotherBuild = "another build"
	// childLeavesNoCheckpoint hands back steps, none of them, and ends well
	// with nothing published. rewardStepsVariable is the variable a starter
	// names the file for the steps by, spelled out here because a starter and
	// a command built apart agree on nothing but the name.
	childLeavesNoCheckpoint = "no checkpoint"
	rewardStepsVariable     = "OVERGO_TRAIN_REWARD_STEPS"
)

// TestMain runs the tests, or, in a child a test started as the training
// command, the command: its own main over the arguments it was started with,
// so the flags a starter renders are the flags the command parses.
func TestMain(m *testing.M) {
	switch os.Getenv(commandChildEnvironment) {
	case "":
		os.Exit(m.Run())
	case childFails:
		fmt.Fprintln(os.Stderr, childFailureLine)
		os.Exit(childFailureStatus)
	case childOfAnotherBuild:
		os.Exit(0)
	case childLeavesNoCheckpoint:
		if err := os.WriteFile(os.Getenv(rewardStepsVariable), []byte("[]"), clioptions.PrivateFileMode); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	case childAnnouncesStoreWait:
		go announceStoreWait()
		fallthrough
	case childStandIn:
		fmt.Fprintln(os.Stderr, childStartedLine)
		denseDevice = stepsOnNoGradient
	}
	main()
	// main returns only when the command succeeded.
	os.Exit(0)
}

// announceStoreWait prints storeWaitLine once the command is inside the wait
// for the store's writer lock. It reads that off the goroutines' stacks: the
// command prints nothing of its own before it has opened the store, and a
// command that did not wait for the lock never gets there.
func announceStoreWait() {
	stacks := make([]byte, 1<<20)
	for !bytes.Contains(stacks[:runtime.Stack(stacks, true)], []byte("processlock.AcquireContext")) {
		runtime.Gosched()
	}
	fmt.Fprintln(os.Stderr, storeWaitLine)
}

// standInLoss is the loss stepsOnNoGradient reports for each group it steps on.
const standInLoss = 0.25

// stepsOnNoGradient stands in for the device trainer: it steps the optimizer
// once for each group with a weight and passes a group with none.
func stepsOnNoGradient(model *densecausal.Model, groups [][]densecausal.WeightedSequence, learningRate, momentum float64, _ bool, resume *densecausal.TrainState, _ densecausal.TrainObserver) ([]float64, int, densecausal.TrainState, error) {
	update, err := model.NewDenseUpdate(learningRate, momentum, resume)
	if err != nil {
		return nil, 0, densecausal.TrainState{}, err
	}
	var losses []float64
	skipped := 0
	for _, group := range groups {
		if !slices.ContainsFunc(group, func(sequence densecausal.WeightedSequence) bool {
			return slices.ContainsFunc(sequence.Weights, func(weight float32) bool { return weight != 0 })
		}) {
			skipped++
			continue
		}
		update.Step()
		losses = append(losses, standInLoss)
	}
	return losses, skipped, update.Snapshot(), nil
}

// rewardCommandFixture is what a workbench's reward run stands on: a store
// that holds an active reward recipe over a tiny model, the model's directory
// and a dataset of one group of answers already scored.
type rewardCommandFixture struct {
	root, storePath, model, dataset string
	store                           *overgodb.Store
	policy, recipe                  artifact.ID
}

func newRewardCommandFixture(t *testing.T) rewardCommandFixture {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	fixture := rewardCommandFixture{root: root, storePath: filepath.Join(root, "store"), model: filepath.Join(root, "model"), dataset: filepath.Join(root, "rollouts.jsonl")}
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 11,
	})
	if err := os.MkdirAll(fixture.model, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(fixture.model, trainingprogram.CheckpointWeights), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "workbench reward evaluator")
	rows := fmt.Sprintf(`{"id":"candidate-a","group":"prompt-a","prompt":"ab","completion":"c","reward":1,"evaluator":"%[1]s"}`+"\n"+
		`{"id":"candidate-b","group":"prompt-a","prompt":"ab","completion":"d","reward":-1,"evaluator":"%[1]s"}`+"\n", evaluator)
	for path, document := range map[string]string{
		filepath.Join(fixture.model, "config.json"):    `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"max_position_embeddings":16,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`,
		filepath.Join(fixture.model, "tokenizer.json"): `{"model":{"type":"BPE","vocab":{"a":1,"b":2,"c":3,"d":4},"merges":[]}}`,
		fixture.dataset: rows,
	} {
		if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	fixture.store = store
	weightsPath, err := trainingworkflow.IdentifyModelWeights(fixture.model)
	if err != nil {
		t.Fatal(err)
	}
	if fixture.policy, err = trainingworkflow.WeightsIdentity(weightsPath); err != nil {
		t.Fatal(err)
	}
	profile := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindProfile, "workbench reward "+name)
	}
	dataset := testutil.ArtifactID(t, artifact.KindDataset, rows)
	split := testutil.ArtifactID(t, artifact.KindDatasetShard, "all\x00"+rows)
	processor := testutil.ArtifactID(t, artifact.KindProfile, "overgo/training/grouped-rollout/v1")
	optimizerPolicy := trainingprogram.BuiltinOptimizerPolicy()
	policies := trainingprogram.PolicySpec{
		Precision: profile("precision"), Placement: profile("placement"), Memory: profile("memory"),
		Checkpoint: profile("checkpoint"), Evaluation: profile("evaluation"), Promotion: profile("promotion"),
		Optimizer: optimizerPolicy.ID,
	}
	loss, evidence := profile("loss"), testutil.ArtifactID(t, artifact.KindEvidence, "workbench reward evidence")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: string(trainingprogram.ObjectiveGRPO), Kind: trainingprogram.ObjectiveGRPO,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: dataset, Split: split, Processors: []artifact.ID{processor}, Loss: loss,
		Evaluation: policies.Evaluation, Metric: trainingprogram.MetricTokenAccuracy,
		Evidence: []artifact.ID{evidence}, Authority: trainingprogram.ObjectiveApproved,
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
	var descriptors []artifact.Descriptor
	for _, id := range []artifact.ID{
		fixture.policy, dataset, split, processor, loss, evidence, evaluator, policies.Precision,
		policies.Placement, policies.Memory, policies.Checkpoint, policies.Evaluation, policies.Promotion,
	} {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/reward-command/authority", Artifacts: descriptors, Contents: []artifact.Content{objectiveContent, optimizerContent}}); err != nil {
		t.Fatal(err)
	}
	dependencies := append([]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: fixture.policy}}, modelrecipetest.PolicyDependencies(policies)...)
	dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyEvaluator, Artifact: evaluator})
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies,
		[]recipe.Node{
			{ID: "batch", Module: workflowrecipe.ModuleBatchRollout, Placement: recipe.PlacementHost},
			{ID: "policy", Module: workflowrecipe.ModuleScorePolicy, Placement: recipe.PlacementHost},
			{ID: "objective", Module: workflowrecipe.ModuleGRPOObjective, Placement: recipe.PlacementHost},
			{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
			{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
		},
		[]recipe.Edge{
			edge("batch", "batch", "policy", "batch"), edge("policy", "scores", "objective", "policy"),
			edge("objective", "loss", "backward", "loss"), edge("backward", "gradients", "optimize", "gradients"),
		}, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/reward-command/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/reward-command/verification", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "reward command fixture"); err != nil {
		t.Fatal(err)
	}
	fixture.recipe = definition.ID
	return fixture
}

// request is the reward run a workbench hands the training command: of one
// step over the fixture's recipe, model and dataset, published under the name
// output, off the host and with its session recorded.
func (fixture rewardCommandFixture) request(output string) trainingworkflow.Request {
	return trainingworkflow.Request{
		Repository: fixture.store, Observations: fixture.store, Recipe: fixture.recipe,
		ModelDirectory: fixture.model, DatasetPath: fixture.dataset, OutputDirectory: filepath.Join(fixture.root, output), Steps: 1,
	}
}

// sessions is the training sessions the store holds under the fixture's
// recipe, read afresh, so that what another process recorded is seen.
func (fixture rewardCommandFixture) sessions(t *testing.T) []runrecord.ServingObservation {
	t.Helper()
	ctx := t.Context()
	if err := fixture.store.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	dependents, err := fixture.store.Children(ctx, fixture.recipe)
	if err != nil {
		t.Fatal(err)
	}
	var sessions []runrecord.ServingObservation
	for _, edge := range dependents {
		dependent, _, err := fixture.store.Artifact(ctx, edge.Child)
		if err != nil {
			t.Fatal(err)
		}
		if dependent.Schema != runrecord.ServingObservationSchema {
			continue
		}
		observation, err := runrecord.RequireServingObservation(ctx, fixture.store, edge.Child)
		if err != nil {
			t.Fatal(err)
		}
		if observation.Task == recipe.TaskTraining {
			sessions = append(sessions, observation)
		}
	}
	return sessions
}

// modelWeights loads the weights of the model a directory holds: the
// fixture's, or the one a run published.
func modelWeights(t *testing.T, directory string) map[string][]float32 {
	t.Helper()
	model, err := densecausal.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	return model.Weights
}

// writeFunc is a writer that is one function.
type writeFunc func([]byte) (int, error)

func (write writeFunc) Write(data []byte) (int, error) { return write(data) }

// TestTrainingCommandRunsAWorkbenchRewardRun holds a reward run a workbench
// starts to running in the training command, a second process, through the
// command's own flags. The child is this test binary run as the command, its
// device trainer a stand-in that steps on no gradient; the store is one this
// process holds open, as a server does.
//
// The run's step comes back to the starter's observer when the child has
// ended, the checkpoint it published loads, the child's output reaches the
// starter's progress, and the store, read afresh, holds a succeeded training
// session under the recipe that this process did not record. A second run
// that names both the model and the checkpoint to resume continues from that
// checkpoint for the steps it asked for, and with no observer its steps are
// in the result.
//
// A run under a recipe the store does not hold comes back with the command's
// status and the command's own refusal, and not with the rest of what the
// child printed. A child that prints a line and ends with a status, writing
// no error of its own, comes back with that status and that line. A child
// that ends well and hands back no steps comes back as a training command of
// another build, and one that hands back steps and publishes no checkpoint as
// the checkpoint that is not there. A command that finds the store's writer
// lock held by another process waits for it and then runs, where one that
// opened the store without waiting would have ended at once. A run whose
// context ends while the child runs comes back as the context's error, so it
// is recorded cancelled. And no run leaves its scratch directory behind.
func TestTrainingCommandRunsAWorkbenchRewardRun(t *testing.T) {
	ctx := t.Context()
	// The starter's scratch goes under a temporary directory of this test's
	// own, so that what a run leaves there can be seen.
	scratch := t.TempDir()
	for _, variable := range []string{"TMP", "TMPDIR"} {
		t.Setenv(variable, scratch)
	}
	fixture := newRewardCommandFixture(t)
	t.Setenv(commandChildEnvironment, childStandIn)
	if before := fixture.sessions(t); len(before) != 0 {
		t.Fatalf("the fixture's store holds %d training sessions before any run", len(before))
	}

	var observed []trainingprogram.GRPOObservation
	var printed bytes.Buffer
	first := fixture.request("first")
	first.ObserveGRPO = func(step trainingprogram.GRPOObservation) { observed = append(observed, step) }
	first.Progress = &printed
	result, err := trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, first)
	if err != nil {
		t.Fatal(err)
	}
	if len(observed) != 1 || observed[0].Step != 1 || observed[0].Loss != standInLoss || !observed[0].Valid() || len(result.GRPO) != 0 {
		t.Fatalf("the run handed its observer %+v and returned %+v; want the one step the child took, handed to the observer alone", observed, result.GRPO)
	}
	if loaded, err := trainingprogram.LoadCheckpoint(first.OutputDirectory); err != nil || loaded.ID() != result.Checkpoint.ID() || result.Checkpoint.Optimizer.Step != 1 {
		t.Fatalf("the run returned checkpoint %s at step %d and %s holds %s (%v); want the checkpoint the child published after one step",
			result.Checkpoint.ID(), result.Checkpoint.Optimizer.Step, first.OutputDirectory, loaded.ID(), err)
	}
	if published, base := modelWeights(t, first.OutputDirectory), modelWeights(t, fixture.model); len(published) == 0 || len(published) != len(base) {
		t.Fatalf("the model the run published holds %d tensors and the one it began at %d; want a model of the same tensors", len(published), len(base))
	}
	if !strings.Contains(printed.String(), "checkpoint="+result.Checkpoint.ID().String()) {
		t.Fatalf("the starter's progress received %q; want what the child printed, its checkpoint among it", printed.String())
	}
	if sessions := fixture.sessions(t); len(sessions) != 1 || sessions[0].Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("after the run the store holds the training sessions %+v; want the one succeeded session the child recorded", sessions)
	}

	// Two steps, where the dataset's one group would make one step of a run
	// that named no count.
	second := fixture.request("second")
	second.ResumeDirectory, second.Steps = first.OutputDirectory, 2
	resumed, err := trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, second)
	if err != nil || len(resumed.GRPO) != 2 || resumed.GRPO[0].Step != 2 || resumed.Checkpoint.Optimizer.Step != 3 {
		t.Fatalf("a run of two steps naming the model and the checkpoint to resume = steps %+v, checkpoint step %d (%v); want it to continue with steps 2 and 3",
			resumed.GRPO, resumed.Checkpoint.Optimizer.Step, err)
	}

	unpublished := fixture.request("unpublished")
	unpublished.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "a recipe the store does not hold")
	if _, err := trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, unpublished); err == nil ||
		!strings.Contains(err.Error(), "ended with status 1") || !strings.Contains(err.Error(), "not a published training recipe") || strings.Contains(err.Error(), childStartedLine) {
		t.Fatalf("a run under a recipe the store does not hold = %v; want the command's status and its own refusal, without the rest of what it printed", err)
	}

	t.Setenv(commandChildEnvironment, childFails)
	want := fmt.Sprintf("training workflow: the training command ended with status %d: %s", childFailureStatus, childFailureLine)
	if _, err := trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, fixture.request("failed")); err == nil || !strings.HasPrefix(err.Error(), want) {
		t.Fatalf("a child that printed a line and ended with a status = %v; want %q, and after it whatever else the child printed", err, want)
	}

	t.Setenv(commandChildEnvironment, childOfAnotherBuild)
	if _, err := trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, fixture.request("silent")); err == nil || !strings.Contains(err.Error(), "training command of another build") {
		t.Fatalf("a child that ended well and handed back no steps = %v; want it named a training command of another build", err)
	}

	t.Setenv(commandChildEnvironment, childLeavesNoCheckpoint)
	if _, err := trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, fixture.request("unpublished-output")); !errors.Is(err, os.ErrNotExist) || strings.Contains(err.Error(), "another build") {
		t.Fatalf("a child that handed back steps and published no checkpoint = %v; want the checkpoint that is not there", err)
	}

	// The store's writer lock is held, as by a process in the middle of a
	// commit, until the child says it is waiting for it.
	t.Setenv(commandChildEnvironment, childAnnouncesStoreWait)
	held, err := processlock.AcquireContext(ctx, fixture.storePath+".lock", clioptions.PrivateFileMode)
	if err != nil {
		t.Fatal(err)
	}
	waiting := processcontrol.WatchLine(nil, storeWaitLine)
	ended, released := make(chan struct{}), make(chan error, 1)
	go func() {
		select {
		case <-waiting.Line():
		case <-ended:
		}
		released <- held.Close()
	}()
	waited := fixture.request("waited")
	waited.Progress = waiting
	_, err = trainingworkflow.ExecuteInCommand(ctx, os.Args[0], fixture.storePath, waited)
	close(ended)
	if releaseErr := <-released; err != nil || releaseErr != nil {
		t.Fatalf("a command started while another process held the store's writer lock = %v (releasing the lock: %v); want it to wait for the lock and then run", err, releaseErr)
	}

	// The cancelled run has a store of its own: a child stopped in the
	// middle of a commit is no part of what the runs above left.
	t.Setenv(commandChildEnvironment, childStandIn)
	interrupted := newRewardCommandFixture(t)
	cancelled, cancel := context.WithCancel(ctx)
	defer cancel()
	stopped := interrupted.request("cancelled")
	stopped.Progress = writeFunc(func(data []byte) (int, error) {
		cancel()
		return len(data), nil
	})
	if _, err := trainingworkflow.ExecuteInCommand(cancelled, os.Args[0], interrupted.storePath, stopped); !errors.Is(err, context.Canceled) {
		t.Fatalf("a run cancelled at the child's first output = %v; want the context's error, so that the run is recorded cancelled", err)
	}

	if left, err := filepath.Glob(filepath.Join(scratch, "overgo-reward-run-*")); err != nil || len(left) != 0 {
		t.Fatalf("the runs left the scratch directories %v behind (%v); want each removed when its run returned", left, err)
	}
}

// TestTrainingCommandRefusesRewardTrainingHeldToTheHost holds the training
// command's -host to what the host trains, which is no reward session. The
// child is this test binary run as the command over its own flags, its device
// trainer the stand-in, so a run that was not refused would train and end
// well.
//
// A run under a reward recipe with -host ends with the workflow's refusal,
// which names the weighted step, publishes nothing at -out, and leaves the
// session it was admitted as recorded as failed. And -pilot with -host is
// refused for the two flags together, before the pilot reads its acceptance:
// a pilot's arms are reward sessions.
func TestTrainingCommandRefusesRewardTrainingHeldToTheHost(t *testing.T) {
	ctx := t.Context()
	fixture := newRewardCommandFixture(t)
	t.Setenv(commandChildEnvironment, childStandIn)
	// refused runs the command over the arguments and returns the status it
	// ended with and the error it wrote for its starter.
	refused := func(arguments ...string) (int, string) {
		t.Helper()
		written := filepath.Join(t.TempDir(), "error.txt")
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{
			Path: os.Args[0], Args: arguments, Env: append(os.Environ(), clioptions.ErrorFileEnvironment+"="+written),
		})
		if err != nil {
			t.Fatal(err)
		}
		// A command that ended well wrote no error.
		refusal, _ := os.ReadFile(written)
		return receipt.ExitCode, string(refusal)
	}
	held := fixture.request("held")
	if status, refusal := refused("-store", fixture.storePath, "-recipe", held.Recipe.String(), "-model", held.ModelDirectory, "-dataset", held.DatasetPath,
		"-out", held.OutputDirectory, "-steps", "1", "-host"); status != 1 || !strings.Contains(refusal, "weighted step") || !strings.Contains(refusal, "held to the host") {
		t.Fatalf("a reward run with -host ended with status %d and %q; want it refused, since a reward session trains through the weighted step and the host has no reward trainer", status, refusal)
	}
	if _, err := os.Stat(held.OutputDirectory); !os.IsNotExist(err) {
		t.Fatalf("a reward run with -host left %s behind (%v); want nothing published by a run that was refused", held.OutputDirectory, err)
	}
	if sessions := fixture.sessions(t); len(sessions) != 1 || sessions[0].Outcome != runrecord.OutcomeFailed {
		t.Fatalf("after the refused run the store holds the training sessions %+v; want the one session the command admitted, recorded as failed", sessions)
	}
	if status, refusal := refused("-store", fixture.storePath, "-pilot", fixture.recipe.String(), "-host"); status != 1 || !strings.Contains(refusal, "-pilot takes no -host") {
		t.Fatalf("-pilot with -host ended with status %d and %q; want the two flags refused together, with the reason", status, refusal)
	}
}
