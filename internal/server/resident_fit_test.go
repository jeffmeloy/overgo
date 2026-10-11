package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// residentFitWorkspace lists the capabilities of each workflow kind and
// counts the runs each one's recipe executed.
type residentFitWorkspace struct {
	capabilities map[WorkflowKind][]WorkflowCapability
	repository   artifact.Repository
	mu           sync.Mutex
	runs         map[artifact.ID]int
}

func (workspace *residentFitWorkspace) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	return workspace.capabilities[kind], nil
}

// ExecuteWorkflow commits a run of its own each time, so every execution
// completes its operation.
func (workspace *residentFitWorkspace) ExecuteWorkflow(
	ctx context.Context,
	_ WorkflowKind,
	capability WorkflowCapability,
	_ json.RawMessage,
	reporter operation.Reporter,
) (operation.Completion, error) {
	workspace.mu.Lock()
	workspace.runs[capability.Recipe]++
	ordinal := workspace.runs[capability.Recipe]
	workspace.mu.Unlock()
	run, err := artifact.IdentifyBytes(artifact.KindRun, fmt.Appendf(nil, "resident fit run %s %d", capability.Recipe, ordinal))
	if err != nil {
		return operation.Completion{}, err
	}
	if _, err := workspace.repository.Commit(ctx, artifact.Batch{
		Key: "fixture/resident-fit/" + run.String(), Artifacts: []artifact.Descriptor{{ID: run}},
	}); err != nil {
		return operation.Completion{Run: run}, err
	}
	reporter.Publishing()
	return operation.Completion{Run: run}, nil
}

func (workspace *residentFitWorkspace) executions(recipeID artifact.ID) int {
	workspace.mu.Lock()
	defer workspace.mu.Unlock()
	return workspace.runs[recipeID]
}

// unreadableDeviceGenerator serves the fake model and cannot read a device's
// memory, as a model served on the host processor cannot.
type unreadableDeviceGenerator struct{ *fakeGenerator }

func (unreadableDeviceGenerator) DeviceMemory(context.Context) (inference.DeviceMemory, error) {
	return inference.DeviceMemory{}, errors.New("fixture: no device serves the model")
}

const (
	// residentFitPeak is the largest peak a succeeded training session of the
	// measured recipe recorded.
	residentFitPeak = 6 * gibibyte
	// residentFitNeed is that peak under the rule: with 15% over it, and 4 GiB.
	residentFitNeed = residentFitPeak*115/100 + 4*gibibyte
	// residentFitSmallerSessions bounds the smaller sessions the fixture
	// records around the largest: each lands on either side of it as its
	// identity falls, so two sides are covered in a handful.
	residentFitSmallerSessions = 64
)

// residentFitFixture is a handler over a store that holds the records below,
// and the recipes its workspace lists.
type residentFitFixture struct {
	handler   *Handler
	workspace *residentFitWorkspace
	// measured is a device training recipe whose succeeded sessions peaked at
	// 6 GiB once and at 2 GiB on the sessions the store lists before and
	// after that one, and which a stage receipt also depends on. host is a
	// host training recipe with the same 6 GiB record. unmeasured is a device
	// training recipe with no record; failed is one whose only session, of
	// 16 GiB, failed, and cancelled one whose only session, of 16 GiB, was
	// cancelled; served is one whose only record, of 16 GiB, is an inference
	// turn's. exported is an export recipe, not a training one, with a 6 GiB
	// training record.
	measured, host, unmeasured, failed, cancelled, served, exported artifact.ID
}

func newResidentFitFixture(t *testing.T, generator Generator) residentFitFixture {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	id := func(kind artifact.Kind, label string) artifact.ID {
		return testutil.ArtifactID(t, kind, "resident fit "+label)
	}
	fixture := residentFitFixture{
		measured: id(artifact.KindRecipe, "measured recipe"), host: id(artifact.KindRecipe, "host recipe"),
		unmeasured: id(artifact.KindRecipe, "unmeasured recipe"), failed: id(artifact.KindRecipe, "failed recipe"),
		cancelled: id(artifact.KindRecipe, "cancelled recipe"),
		served:    id(artifact.KindRecipe, "served recipe"), exported: id(artifact.KindRecipe, "exported recipe"),
	}
	model, environment := id(artifact.KindModel, "model"), id(artifact.KindEvidence, "environment")
	staged := id(artifact.KindEvidence, "staged operation")
	authorities := []artifact.Descriptor{{ID: model}, {ID: environment}, {ID: staged}}
	workspace := &residentFitWorkspace{repository: store, runs: map[artifact.ID]int{}, capabilities: map[WorkflowKind][]WorkflowCapability{}}
	for _, recipeID := range []artifact.ID{
		fixture.measured, fixture.host, fixture.unmeasured, fixture.failed, fixture.cancelled, fixture.served, fixture.exported,
	} {
		authorities = append(authorities, artifact.Descriptor{ID: recipeID})
		kind := WorkflowTraining
		if recipeID == fixture.exported {
			kind = workflowExport
		}
		workspace.capabilities[kind] = append(workspace.capabilities[kind], WorkflowCapability{
			Task: recipe.TaskTraining, Recipe: recipeID, Host: recipeID == fixture.host,
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "train", Module: "test.train"}}},
		})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "resident-fit/authorities", Artifacts: authorities}); err != nil {
		t.Fatal(err)
	}
	started := int64(0)
	record := func(recipeID artifact.ID, task recipe.Task, outcome runrecord.Outcome, failure string, peak uint64) artifact.ID {
		t.Helper()
		started++
		published, err := runrecord.PublishServingObservation(ctx, store, runrecord.ServingObservation{
			Model: model, Recipe: recipeID, Environment: environment, Task: task, Outcome: outcome, Failure: failure,
			StartedUnixNS: started, MeasuredNS: 1, Resources: runrecord.ServingResources{PeakDeviceBytes: peak},
		})
		if err != nil {
			t.Fatal(err)
		}
		return published.ID
	}
	largest := record(fixture.measured, recipe.TaskTraining, runrecord.OutcomeSucceeded, "", residentFitPeak)
	// Smaller sessions are recorded until the store lists one before the
	// largest and one after it, so neither the first nor the last record
	// read can stand for the largest. A record's place among its recipe's
	// dependents does not move when another is added.
	for before, after, sessions := false, false, 0; !before || !after; sessions++ {
		if sessions == residentFitSmallerSessions {
			t.Fatalf("fixture: %d smaller sessions were all listed on one side of the largest", sessions)
		}
		smaller := record(fixture.measured, recipe.TaskTraining, runrecord.OutcomeSucceeded, "", 2*gibibyte)
		dependents, err := store.Children(ctx, fixture.measured)
		if err != nil {
			t.Fatal(err)
		}
		at := func(id artifact.ID) int {
			return slices.IndexFunc(dependents, func(edge artifact.Lineage) bool { return edge.Child == id })
		}
		before, after = before || at(smaller) < at(largest), after || at(smaller) > at(largest)
	}
	record(fixture.host, recipe.TaskTraining, runrecord.OutcomeSucceeded, "", residentFitPeak)
	record(fixture.failed, recipe.TaskTraining, runrecord.OutcomeFailed, "execution_failed", 16*gibibyte)
	record(fixture.cancelled, recipe.TaskTraining, runrecord.OutcomeCancelled, "", 16*gibibyte)
	record(fixture.served, recipe.TaskInference, runrecord.OutcomeSucceeded, "", 16*gibibyte)
	record(fixture.exported, recipe.TaskTraining, runrecord.OutcomeSucceeded, "", residentFitPeak)
	// A record that depends on the measured recipe and is no serving
	// observation: the reader passes it by.
	if _, err := runrecord.PublishStageReceipt(ctx, store, runrecord.StageReceipt{
		Recipe: fixture.measured, Node: "train", Operation: staged, Attempt: 1, State: runrecord.StageAdmitted,
	}, nil, nil); err != nil {
		t.Fatal(err)
	}
	fixture.workspace = workspace
	fixture.handler, err = New(Config{Repository: store, Workflows: workspace}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = fixture.handler.Close() })
	return fixture
}

// post submits the recipe to a workflow route, numbered when request is set.
func (fixture residentFitFixture) post(t *testing.T, path string, recipeID artifact.ID, request string) (int, string, artifact.ID) {
	t.Helper()
	response := serveTestRequest(fixture.handler, http.MethodPost, path, mustJSON(t, workflowRequest{
		Task: recipe.TaskTraining, Recipe: recipeID, Input: json.RawMessage(`{}`), Request: request,
	}))
	var submission workflowResponse
	if response.Code == http.StatusAccepted {
		if err := json.Unmarshal(response.Body.Bytes(), &submission); err != nil {
			t.Fatal(err)
		}
	}
	return response.Code, response.Body.String(), submission.Operation
}

// admitted requires a run of the recipe, an export of the exported recipe and
// a training run of any other, to be accepted and its operation to complete,
// and returns the operation.
func (fixture residentFitFixture) admitted(t *testing.T, recipeID artifact.ID, request, why string) artifact.ID {
	t.Helper()
	path := "/training/run"
	if recipeID == fixture.exported {
		path = "/export/run"
	}
	code, body, id := fixture.post(t, path, recipeID, request)
	if code != http.StatusAccepted {
		t.Fatalf("%s: status=%d body=%s, want the run accepted", why, code, body)
	}
	waited := serveTestRequest(fixture.handler, http.MethodGet, "/operations/wait?id="+id.String(), "")
	var status operation.Status
	if err := json.Unmarshal(waited.Body.Bytes(), &status); err != nil || waited.Code != http.StatusOK || status.State != operation.StateCompleted {
		t.Fatalf("%s: wait status=%d body=%s (%v), want a completed operation", why, waited.Code, waited.Body.String(), err)
	}
	return id
}

// refused requires a training run of the measured recipe to be refused as
// one that does not fit, naming the measured peak, what it needs free and
// what is free, and saying whether it would fit with the served model
// released, and to have run nothing.
func (fixture residentFitFixture) refused(t *testing.T, request string, free uint64, released, why string) {
	t.Helper()
	before := fixture.workspace.executions(fixture.measured)
	code, body, _ := fixture.post(t, "/training/run", fixture.measured, request)
	if code != http.StatusServiceUnavailable || !strings.Contains(body, "insufficient_memory") {
		t.Fatalf("%s: status=%d body=%s, want a 503 insufficient_memory refusal", why, code, body)
	}
	for name, bytes := range map[string]uint64{"measured peak": residentFitPeak, "need": residentFitNeed, "free bytes": free} {
		if !strings.Contains(body, strconv.FormatUint(bytes, 10)) {
			t.Fatalf("%s: the refusal does not name the %s %d: %s", why, name, bytes, body)
		}
	}
	if !strings.Contains(body, "with the served model released it "+released) {
		t.Fatalf("%s: the refusal does not say that with the served model released it %s: %s", why, released, body)
	}
	if got := fixture.workspace.executions(fixture.measured); got != before {
		t.Fatalf("%s: a refused run executed (%d runs, %d before)", why, got, before)
	}
}

// TestResidentFitAdmitsADeviceRunOnlyBesideTheServedModel: a device training
// run is admitted only when its measured peak fits beside the served model. A
// recipe whose succeeded training sessions peaked at 6 GiB at most needs
// 10.9 GiB free: at 10 GiB free the run is refused with those numbers and
// nothing runs, while the command that reproduces it is still answered; at
// 20 GiB free it runs. The refusal says whether the run would fit with the
// served model's 12 GiB released, and that it would not on a device another
// process fills. A numbered submission that started is answered with its
// operation when it is repeated after its own run has taken the memory, and a
// numbered submission that was refused runs under the same number once memory
// is free. On the full device a host run, a run no succeeded training session
// measured (no session, a failed one, a cancelled one, or only an inference
// turn) and a workflow that is no training run are admitted, as are a
// measured run on a server whose model is on no device and on one that cannot
// read its device's memory.
func TestResidentFitAdmitsADeviceRunOnlyBesideTheServedModel(t *testing.T) {
	t.Parallel()
	free := func(bytes uint64) inference.DeviceMemory {
		memory := testDeviceMemory()
		memory.FreeBytes = bytes
		return memory
	}
	const fits, neither = "would fit", "would not fit either"
	generator := &memoryOwnersGenerator{fakeGenerator: &fakeGenerator{}, memory: free(10 * gibibyte)}
	fixture := newResidentFitFixture(t, generator)

	fixture.refused(t, "", 10*gibibyte, fits, "a 6 GiB run at 10 GiB free")
	if code, body, _ := fixture.post(t, "/training/command", fixture.measured, ""); code != http.StatusOK {
		t.Fatalf("the command for a run that does not fit: status=%d body=%s, want it answered", code, body)
	}

	// A device another process fills: 1 GiB free, and no served bytes to release.
	generator.set(inference.DeviceMemory{TotalBytes: 24 * gibibyte, FreeBytes: 1 * gibibyte})
	fixture.refused(t, "", 1*gibibyte, neither, "a 6 GiB run on a full device")
	fixture.admitted(t, fixture.host, "", "a host run on a full device")
	fixture.admitted(t, fixture.unmeasured, "", "a device run no session measured, on a full device")
	fixture.admitted(t, fixture.failed, "", "a device run whose only session failed, on a full device")
	fixture.admitted(t, fixture.cancelled, "", "a device run whose only session was cancelled, on a full device")
	fixture.admitted(t, fixture.served, "", "a device run whose only record is an inference turn, on a full device")
	fixture.admitted(t, fixture.exported, "", "an export under a recipe with a 6 GiB training record, on a full device")

	generator.set(free(20 * gibibyte))
	fixture.admitted(t, fixture.measured, "", "a 6 GiB run at 20 GiB free")
	if got := fixture.workspace.executions(fixture.measured); got != 1 {
		t.Fatalf("%d runs of the measured recipe after one admitted run, want one", got)
	}

	// A numbered run that started is not judged again against the memory its
	// own run now holds: the repeat names the operation the first started.
	first := fixture.admitted(t, fixture.measured, "fit-retry", "a numbered 6 GiB run at 20 GiB free")
	generator.set(free(1 * gibibyte))
	if again := fixture.admitted(t, fixture.measured, "fit-retry", "its repeat at 1 GiB free"); again != first {
		t.Fatalf("the repeat named operation %s, want the first's %s", again, first)
	}
	if got := fixture.workspace.executions(fixture.measured); got != 2 {
		t.Fatalf("%d runs of the measured recipe after one more numbered run and its repeat, want two", got)
	}

	// A refused numbered run is forgotten, so its number runs once memory is free.
	fixture.refused(t, "fit-refused", 1*gibibyte, fits, "a numbered 6 GiB run at 1 GiB free")
	generator.set(free(20 * gibibyte))
	fixture.admitted(t, fixture.measured, "fit-refused", "the refused number at 20 GiB free")

	// The need is exact: a byte under it is refused, the need itself admitted.
	generator.set(free(residentFitNeed - 1))
	fixture.refused(t, "", residentFitNeed-1, fits, "a 6 GiB run a byte short of its need")
	generator.set(free(residentFitNeed))
	fixture.admitted(t, fixture.measured, "", "a 6 GiB run at exactly its need")

	hostServed := newResidentFitFixture(t, &fakeGenerator{})
	hostServed.admitted(t, hostServed.measured, "", "a 6 GiB run beside a model on no device")
	unreadable := newResidentFitFixture(t, unreadableDeviceGenerator{&fakeGenerator{}})
	unreadable.admitted(t, unreadable.measured, "", "a 6 GiB run on a server that cannot read its device's memory")
}

// TestResidentFitNeedsThePeakWithItsMarginAndReserveFree: the rule's
// arithmetic on a 24 GiB device with 10 GiB free beside a 12 GiB served
// model. A run needs its measured peak times 115/100, plus 4 GiB, in whole
// bytes: 5 GiB fits beside the model, 6 GiB fits only with the model
// released, 16 GiB fits neither way, and a need equal to what is free, or to
// what the model's release would free, fits where a byte more does not. A
// peak whose margin cannot be counted is refused, where a product that
// wrapped would have fitted.
func TestResidentFitNeedsThePeakWithItsMarginAndReserveFree(t *testing.T) {
	t.Parallel()
	const (
		// countable is the largest peak whose margin a count still holds;
		// one byte more wraps to a product of 109, a need of 4 GiB and a byte.
		countable = math.MaxUint64 / 115
		reserve   = 4 * gibibyte
		// large is what a 16 GiB peak needs, and served the model's bytes.
		large  = 16*gibibyte*115/100 + reserve
		served = 12 * gibibyte
	)
	device := testDeviceMemory()
	exact, short, released, held := device, device, device, device
	exact.FreeBytes, short.FreeBytes = residentFitNeed, residentFitNeed-1
	released.FreeBytes, held.FreeBytes = large-served, large-served-1
	for _, row := range []struct {
		name          string
		required      uint64
		memory        inference.DeviceMemory
		need          uint64
		beside, alone bool
	}{
		{"5 GiB beside the served model", 5 * gibibyte, device, 5*gibibyte*115/100 + reserve, true, true},
		{"6 GiB only with the model released", residentFitPeak, device, residentFitNeed, false, true},
		{"16 GiB neither way", 16 * gibibyte, device, large, false, false},
		{"a need equal to what is free", residentFitPeak, exact, residentFitNeed, true, true},
		{"a need a byte over what is free", residentFitPeak, short, residentFitNeed, false, true},
		{"a need equal to what the model's release would free", 16 * gibibyte, released, large, false, true},
		{"a need a byte over what the model's release would free", 16 * gibibyte, held, large, false, false},
		{"a peak of nothing needs the reserve alone", 0, device, reserve, true, true},
		{"the largest countable peak", countable, device, countable*115/100 + reserve, false, false},
		{"a peak a byte past counting", countable + 1, device, math.MaxUint64, false, false},
		{"the largest peak", math.MaxUint64, device, math.MaxUint64, false, false},
	} {
		need, beside, alone := residentFit(row.required, row.memory)
		if need != row.need || beside != row.beside || alone != row.alone {
			t.Errorf("%s: need %d, beside %t, alone %t; want need %d, beside %t, alone %t",
				row.name, need, beside, alone, row.need, row.beside, row.alone)
		}
	}
}
