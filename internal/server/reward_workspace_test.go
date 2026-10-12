package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/densecausal"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// rewardTrainer stands in for the training command a server starts a reward
// run in: it keeps each request it is handed and runs it in this process,
// with a device trainer that steps the optimizer once for each group that
// has a weight, on no gradient.
type rewardTrainer struct {
	mu       sync.Mutex
	requests []trainingworkflow.Request
}

func (trainer *rewardTrainer) run(ctx context.Context, request trainingworkflow.Request) (trainingworkflow.Result, error) {
	trainer.mu.Lock()
	trainer.requests = append(trainer.requests, request)
	trainer.mu.Unlock()
	request.Device = func(model *densecausal.Model, groups [][]densecausal.WeightedSequence, learningRate, momentum float64, _ bool, resume *densecausal.TrainState, _ densecausal.TrainObserver) ([]float64, int, densecausal.TrainState, error) {
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
			losses = append(losses, 0.25)
		}
		return losses, skipped, update.Snapshot(), nil
	}
	return trainingworkflow.Execute(ctx, request)
}

// handed is the requests the trainer has been handed so far.
func (trainer *rewardTrainer) handed() []trainingworkflow.Request {
	trainer.mu.Lock()
	defer trainer.mu.Unlock()
	return slices.Clone(trainer.requests)
}

// metricReporter keeps the name of each metric a run reports.
type metricReporter struct {
	testReporter
	mu      sync.Mutex
	metrics []string
}

func (reporter *metricReporter) Metric(metric operation.Metric) {
	reporter.mu.Lock()
	reporter.metrics = append(reporter.metrics, metric.Name)
	reporter.mu.Unlock()
}

// rewardTrainingFixture: a tiny reward program activated in a fresh store,
// its training workspace over the fixture roots, the policy's directory, and
// the dataset a run trains on, one group of two answers already scored.
type rewardTrainingFixture struct {
	store                   *overgodb.Store
	roots                   dataroot.Roots
	workspace               *trainingWorkspace
	definition              recipe.Definition
	dataset, policy         artifact.ID
	policyPath, datasetPath string
}

// newRewardFixture activates the reward program and opens its workspace
// with rewards as the way a reward run executes, nil for a server started
// with no training command.
func newRewardFixture(t *testing.T, rewards func(context.Context, trainingworkflow.Request) (trainingworkflow.Result, error)) rewardTrainingFixture {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	roots := dataroot.Roots{Models: filepath.Join(root, "models"), Datasets: filepath.Join(root, "datasets"), Checkpoints: filepath.Join(root, "checkpoints")}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 11,
	})
	fixture := rewardTrainingFixture{store: store, roots: roots, policyPath: filepath.Join(roots.Models, "policy"), datasetPath: filepath.Join(roots.Datasets, "rollouts.jsonl")}
	fixture.policy = writeWorkspaceModel(t, fixture.policyPath, weights, shapes)
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "reward workspace evaluator")
	rows := fmt.Sprintf(`{"id":"candidate-a","group":"prompt-a","prompt":"ab","completion":"c","reward":1,"evaluator":"%[1]s"}`+"\n"+
		`{"id":"candidate-b","group":"prompt-a","prompt":"ab","completion":"d","reward":-1,"evaluator":"%[1]s"}`+"\n", evaluator)
	if err := os.MkdirAll(roots.Datasets, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fixture.datasetPath, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	fixture.dataset = testutil.ArtifactID(t, artifact.KindDataset, rows)
	split := testutil.ArtifactID(t, artifact.KindDatasetShard, "all\x00"+rows)
	profile := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindProfile, "reward workspace "+name)
	}
	processor := testutil.ArtifactID(t, artifact.KindProfile, "overgo/training/grouped-rollout/v1")
	optimizerPolicy := trainingprogram.BuiltinOptimizerPolicy()
	policies := trainingprogram.PolicySpec{
		Precision: profile("precision"), Placement: profile("placement"), Memory: profile("memory"),
		Checkpoint: profile("checkpoint"), Evaluation: profile("evaluation"), Promotion: profile("promotion"),
		Optimizer: optimizerPolicy.ID,
	}
	loss, evidence := profile("loss"), testutil.ArtifactID(t, artifact.KindEvidence, "reward workspace objective evidence")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: string(trainingprogram.ObjectiveGRPO), Kind: trainingprogram.ObjectiveGRPO,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: fixture.dataset, Split: split, Processors: []artifact.ID{processor}, Loss: loss,
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
	descriptors := []artifact.Descriptor{{ID: fixture.policy, Size: 1}, {ID: fixture.dataset, Size: uint64(len(rows))}}
	for _, id := range []artifact.ID{
		split, processor, loss, evidence, evaluator, policies.Precision, policies.Placement,
		policies.Memory, policies.Checkpoint, policies.Evaluation, policies.Promotion,
	} {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/reward-workspace/artifacts", Artifacts: descriptors,
		Contents: []artifact.Content{objectiveContent, optimizerContent},
		Locations: []artifact.LocationEvent{
			location(fixture.policy, artifact.LocationDirectory, fixture.policyPath),
			location(fixture.dataset, artifact.LocationFile, fixture.datasetPath),
		},
	}); err != nil {
		t.Fatal(err)
	}
	dependencies := append([]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: fixture.policy}}, modelrecipetest.PolicyDependencies(policies)...)
	fixture.definition, err = trainingWorkspaceAdmissionDefinition(trainingprogram.ObjectiveGRPO,
		append(dependencies, recipe.Dependency{Role: recipe.DependencyEvaluator, Artifact: evaluator}))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/reward-workspace/candidate", fixture.definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/reward-workspace/verification", fixture.definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, fixture.definition, verification, recipe.EvidenceVerified, "reward workspace fixture"); err != nil {
		t.Fatal(err)
	}
	if fixture.workspace, err = NewTrainingWorkspace(ctx, store, roots, fixture.policy, rewards); err != nil {
		t.Fatal(err)
	}
	return fixture
}

// input is a run of so many steps over the fixture's dataset, published
// under the name output.
func (fixture rewardTrainingFixture) input(t *testing.T, output string, steps int) json.RawMessage {
	t.Helper()
	input, err := json.Marshal(map[string]any{"dataset": fixture.dataset, "output": output, "steps": steps})
	if err != nil {
		t.Fatal(err)
	}
	return input
}

// trainerAbsent is from the reason a server started without the training
// command gives for running no reward session.
const trainerAbsent = "the training command, which this server was started without"

// sessionDevices is the device each training session the store holds under
// the recipe was recorded on: cpu for a session held to the host, and the
// graphics card's name for one that was not.
func sessionDevices(t *testing.T, store *overgodb.Store, recipeID artifact.ID) []string {
	t.Helper()
	ctx := t.Context()
	dependents, err := store.Children(ctx, recipeID)
	if err != nil {
		t.Fatal(err)
	}
	var devices []string
	for _, edge := range dependents {
		dependent, _, err := store.Artifact(ctx, edge.Child)
		if err != nil {
			t.Fatal(err)
		}
		if dependent.Schema != runrecord.ServingObservationSchema {
			continue
		}
		observation, err := runrecord.RequireServingObservation(ctx, store, edge.Child)
		if err != nil {
			t.Fatal(err)
		}
		if observation.Task != recipe.TaskTraining {
			continue
		}
		environment, err := runrecord.RequireEnvironment(ctx, store, observation.Environment)
		if err != nil {
			t.Fatal(err)
		}
		devices = append(devices, environment.Device)
	}
	return devices
}

// TestTrainingWorkspaceTrainsRewardsInTheTrainingCommand holds a reward run
// from the workbench to leaving the server's process.
//
// The reward capability does not say it trains on the host processor: it is
// listed with no refusal and with the note that it trains on the device, in
// the training command's own process. A run is handed to the way the server
// was started to run one, here a stand-in that keeps the request and runs it
// in this process. The request names the recipe, the policy's directory, the
// dataset, the output under the checkpoints root and the steps, and is not
// held to the host; and it is one the training command's flags carry: handed
// to the command's route with no command at the path, it gets as far as
// starting the command. The run publishes the checkpoint it trained, a reward
// trace of its one step and a decision that waits for an evaluation; its
// session is recorded as a device session; and the step it reports carries no
// gradient or update norm, which a device step keeps on the device.
//
// A server started with no training command lists the capability with the
// reason, and a run is refused with that reason, recorded as a failed run,
// and trains nothing.
//
// A preference run is not handed over. Its capability still says it trains on
// the host processor, with no note and no refusal, and its session is
// recorded as a host session, which holds no device memory whatever the
// server's process holds.
func TestTrainingWorkspaceTrainsRewardsInTheTrainingCommand(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	trainer := &rewardTrainer{}
	fixture := newRewardFixture(t, trainer.run)
	capabilities, err := fixture.workspace.WorkflowCapabilities(ctx, WorkflowTraining)
	if err != nil || len(capabilities) != 1 || capabilities[0].Recipe != fixture.definition.ID {
		t.Fatalf("capabilities=%+v err=%v", capabilities, err)
	}
	if listed := capabilities[0]; listed.Host || listed.Refusal != "" || !strings.Contains(listed.Note, "in the training command's own process") {
		t.Fatalf("the reward capability is listed with host %t, refusal %q and note %q; want a run off the host, no refusal, and the note that it trains in the training command's own process",
			listed.Host, listed.Refusal, listed.Note)
	}
	reporter := &metricReporter{}
	completion, err := fixture.workspace.ExecuteWorkflow(ctx, WorkflowTraining, capabilities[0], fixture.input(t, "trained", 1), reporter)
	if err != nil || completion.Run.Kind() != artifact.KindRun || len(completion.Outputs) != 3 {
		t.Fatalf("completion=%+v err=%v, want a run with a checkpoint, a trace and a decision", completion, err)
	}
	handed, output := trainer.handed(), filepath.Join(fixture.roots.Checkpoints, "trained")
	if len(handed) != 1 || handed[0].Recipe != fixture.definition.ID || handed[0].ModelDirectory != fixture.policyPath ||
		handed[0].DatasetPath != fixture.datasetPath || handed[0].OutputDirectory != output || handed[0].Steps != 1 || handed[0].Host {
		t.Fatalf("the run was handed over as %+v; want one request for recipe %s over %s and %s to %s, of one step, not held to the host",
			handed, fixture.definition.ID, fixture.policyPath, fixture.datasetPath, output)
	}
	absent := filepath.Join(t.TempDir(), "absent-train.exe")
	if _, err := trainingworkflow.ExecuteInCommand(ctx, absent, filepath.Join(t.TempDir(), "store"), handed[0]); err == nil || !strings.Contains(err.Error(), "start "+absent) {
		t.Fatalf("the request the workspace hands over, given to the training command's route with no command at %s = %v; want it carried as far as starting the command", absent, err)
	}
	if checkpoint, err := trainingprogram.LoadCheckpoint(output); err != nil || checkpoint.ID() != completion.Outputs[0] {
		t.Fatalf("checkpoint=%s err=%v, want the run's %s", checkpoint.ID(), err, completion.Outputs[0])
	}
	var trace runrecord.TrainingTrace
	readJSON(t, fixture.store, completion.Outputs[1], &trace)
	if trace.Run != completion.Run || trace.Recipe != fixture.definition.ID || trace.Dataset != fixture.dataset || trace.Policy != fixture.policy ||
		trace.Objective != trainingprogram.ObjectiveGRPO || len(trace.GRPO) != 1 {
		t.Fatalf("trace=%+v, want a reward trace of the run's one step", trace)
	}
	var decision runrecord.TrainingDecision
	readJSON(t, fixture.store, completion.Outputs[2], &decision)
	if decision.State != runrecord.TrainingEvaluationRequired || decision.Run != completion.Run || decision.Parent != fixture.policy ||
		decision.Checkpoint != completion.Outputs[0] || decision.Trace != completion.Outputs[1] {
		t.Fatalf("decision=%+v, want one that waits for an evaluation of the run's checkpoint", decision)
	}
	if reported := reporter.metrics; !slices.Contains(reported, "grpo_loss") || slices.Contains(reported, "gradient_l2") || slices.Contains(reported, "update_l2") {
		t.Fatalf("the run reported the metrics %v; want its step's loss and no gradient or update norm", reported)
	}
	if devices := sessionDevices(t, fixture.store, fixture.definition.ID); !slices.Equal(devices, []string{"cuda0"}) {
		t.Fatalf("the reward run's session was recorded on %v; want one session, on the device", devices)
	}

	without := newRewardFixture(t, nil)
	refused, err := without.workspace.WorkflowCapabilities(ctx, WorkflowTraining)
	if err != nil || len(refused) != 1 || refused[0].Host || !strings.Contains(refused[0].Refusal, trainerAbsent) {
		t.Fatalf("with no training command the capability is listed as %+v (%v); want it off the host and refused for the missing command", refused, err)
	}
	completion, err = without.workspace.ExecuteWorkflow(ctx, WorkflowTraining, refused[0], without.input(t, "trained", 1), testReporter{})
	if err == nil || !strings.Contains(err.Error(), refused[0].Refusal) || completion.Run.Kind() != artifact.KindRun {
		t.Fatalf("with no training command a run = %+v, %v; want it refused with the listed reason and recorded as a failed run", completion, err)
	}
	if _, err := os.Stat(filepath.Join(without.roots.Checkpoints, "trained")); !os.IsNotExist(err) {
		t.Fatalf("a refused run left a checkpoint: %v", err)
	}

	preference := newDPOTrainingFixture(t)
	hosted, err := preference.workspace.WorkflowCapabilities(ctx, WorkflowTraining)
	if err != nil || len(hosted) != 1 || !hosted[0].Host || hosted[0].Note != "" || hosted[0].Refusal != "" {
		t.Fatalf("the preference capability is listed as %+v (%v); want it on the host processor with no note and no refusal", hosted, err)
	}
	input, err := json.Marshal(map[string]any{"dataset": preference.dataset, "output": "trained", "steps": 1, "objective_scale": 0.1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := preference.workspace.ExecuteWorkflow(ctx, WorkflowTraining, hosted[0], input, testReporter{}); err != nil {
		t.Fatal(err)
	}
	if devices := sessionDevices(t, preference.store, preference.definition.ID); !slices.Equal(devices, []string{"cpu"}) {
		t.Fatalf("the preference run's session was recorded on %v; want one session, on the host processor", devices)
	}
}

// TestRewardRunIsWeighedBesideTheServedModel holds a reward run from the
// workbench to the rule a device run is admitted by.
//
// Its capability does not train on the host, so its recipe's measured peak is
// weighed against what the device has free beside the served model. A recipe
// whose succeeded training session peaked at 6 GiB needs 10.9 GiB free: at
// 10 GiB free the run is refused with those numbers and is never handed to
// the trainer, and at 20 GiB free it is handed over and completes.
//
// A server started with no training command answers the same run, at 10 GiB
// free, with the reason reward training cannot run: the run is accepted and
// fails with that reason, where weighing it would have said it does not fit.
func TestRewardRunIsWeighedBesideTheServedModel(t *testing.T) {
	t.Parallel()
	// served records a succeeded training session of the fixture's recipe that
	// peaked at 6 GiB on the device, and serves the fixture's workspace beside
	// a model on a device with 10 GiB free.
	served := func(fixture rewardTrainingFixture) (*Handler, *memoryOwnersGenerator) {
		t.Helper()
		environment := testutil.ArtifactID(t, artifact.KindEvidence, "reward workspace device environment")
		if _, err := fixture.store.Commit(t.Context(), artifact.Batch{Key: "fixture/reward-workspace/environment", Artifacts: []artifact.Descriptor{{ID: environment}}}); err != nil {
			t.Fatal(err)
		}
		if _, err := runrecord.PublishServingObservation(t.Context(), fixture.store, runrecord.ServingObservation{
			Model: fixture.policy, Recipe: fixture.definition.ID, Environment: environment, Task: recipe.TaskTraining, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: 1, MeasuredNS: 1, Resources: runrecord.ServingResources{PeakDeviceBytes: residentFitPeak},
		}); err != nil {
			t.Fatal(err)
		}
		generator := &memoryOwnersGenerator{fakeGenerator: &fakeGenerator{}, memory: testDeviceMemory()}
		handler, err := New(Config{Repository: fixture.store, Workflows: fixture.workspace}, generator)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = handler.Close() })
		return handler, generator
	}
	// submit posts a run of one step, and for an accepted one waits for how
	// its operation ended.
	submit := func(handler *Handler, fixture rewardTrainingFixture, output string) (int, string, operation.Status) {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/training/run", mustJSON(t, workflowRequest{
			Task: recipe.TaskTraining, Recipe: fixture.definition.ID, Input: fixture.input(t, output, 1),
		}))
		var status operation.Status
		if response.Code == http.StatusAccepted {
			var submission workflowResponse
			if err := json.Unmarshal(response.Body.Bytes(), &submission); err != nil {
				t.Fatal(err)
			}
			waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
			if err := json.Unmarshal(waited.Body.Bytes(), &status); err != nil || waited.Code != http.StatusOK {
				t.Fatalf("wait status=%d body=%s (%v)", waited.Code, waited.Body.String(), err)
			}
		}
		return response.Code, response.Body.String(), status
	}

	trainer := &rewardTrainer{}
	fixture := newRewardFixture(t, trainer.run)
	handler, generator := served(fixture)
	code, body, _ := submit(handler, fixture, "refused")
	if code != http.StatusServiceUnavailable || !strings.Contains(body, "insufficient_memory") {
		t.Fatalf("a reward run that measured 6 GiB, at 10 GiB free: status=%d body=%s; want a 503 insufficient_memory refusal", code, body)
	}
	for name, bytes := range map[string]uint64{"measured peak": residentFitPeak, "need": residentFitNeed, "free bytes": 10 * gibibyte} {
		if !strings.Contains(body, strconv.FormatUint(bytes, 10)) {
			t.Fatalf("the refusal does not name the %s %d: %s", name, bytes, body)
		}
	}
	if handed := trainer.handed(); len(handed) != 0 {
		t.Fatalf("a refused run was handed to the trainer: %+v", handed)
	}
	memory := testDeviceMemory()
	memory.FreeBytes = 20 * gibibyte
	generator.set(memory)
	if code, body, status := submit(handler, fixture, "trained"); code != http.StatusAccepted || status.State != operation.StateCompleted || len(trainer.handed()) != 1 {
		t.Fatalf("the same run at 20 GiB free: status=%d body=%s, operation %+v, %d runs handed to the trainer; want it accepted, handed over once and completed",
			code, body, status, len(trainer.handed()))
	}

	without := newRewardFixture(t, nil)
	absent, _ := served(without)
	if code, body, status := submit(absent, without, "trained"); code != http.StatusAccepted || status.State != operation.StateFailed || !strings.Contains(status.Failure, trainerAbsent) {
		t.Fatalf("with no training command, the run at 10 GiB free: status=%d body=%s, operation %+v; want it accepted and failed for the missing command, not weighed",
			code, body, status)
	}
}
