package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestWorkflowRunIsIdempotentByRequestID posts one numbered submission many times,
// at once and again afterwards: every answer names one operation and the executor
// ran once; the same number on other input is refused, a number that is no token is
// refused, and a submission with no number still starts an operation of its own each
// time.
func TestWorkflowRunIsIdempotentByRequestID(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "numbered workflow recipe")
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{}, repository: store,
		run: testutil.ArtifactID(t, artifact.KindRun, "numbered workflow run"),
		capability: WorkflowCapability{
			Task: recipe.TaskGeneration, Recipe: recipeID,
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []workflowControl{{Name: "text", Type: workflowControlText, Required: true}},
		},
	}
	handler, err := New(Config{Repository: store, Workflows: generator}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	post := func(request, text string) (int, workflowResponse) {
		t.Helper()
		body, err := json.Marshal(workflowRequest{Task: recipe.TaskGeneration, Recipe: recipeID, Input: json.RawMessage(fmt.Sprintf(`{"text":%q}`, text)), Request: request})
		if err != nil {
			t.Fatal(err)
		}
		response := serveTestRequest(handler, http.MethodPost, "/generation/run", string(body))
		var submission workflowResponse
		if response.Code == http.StatusAccepted {
			if err := json.Unmarshal(response.Body.Bytes(), &submission); err != nil {
				t.Fatal(err)
			}
		}
		return response.Code, submission
	}
	executions := func() int {
		generator.mu.Lock()
		defer generator.mu.Unlock()
		return generator.executions
	}

	// Eight posts of one numbered submission at once, then one more after it finished.
	operations := make([]artifact.ID, 8)
	var group sync.WaitGroup
	for index := range operations {
		group.Go(func() {
			code, submission := post("run-one", "hello")
			if code != http.StatusAccepted {
				t.Errorf("post %d answered %d", index, code)
			}
			operations[index] = submission.Operation
		})
	}
	group.Wait()
	for _, id := range operations {
		if id != operations[0] || !id.Valid() {
			t.Fatalf("numbered posts named operations %v, want one", operations)
		}
	}
	if waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+operations[0].String(), ""); waited.Code != http.StatusOK {
		t.Fatalf("wait status=%d body=%s", waited.Code, waited.Body.String())
	}
	if code, again := post("run-one", "hello"); code != http.StatusAccepted || again.Operation != operations[0] {
		t.Fatalf("a repeat after the run finished = (%d, %v), want the same operation %v", code, again.Operation, operations[0])
	}
	if got := executions(); got != 1 {
		t.Fatalf("%d executions of one numbered submission, want one", got)
	}

	// The number belongs to what was asked: other input under it is refused, and starts nothing.
	response := serveTestRequest(handler, http.MethodPost, "/generation/run", mustJSON(t, workflowRequest{Task: recipe.TaskGeneration, Recipe: recipeID, Input: json.RawMessage(`{"text":"another"}`), Request: "run-one"}))
	if response.Code != http.StatusConflict || !strings.Contains(response.Body.String(), "request_id_reused") {
		t.Fatalf("reused number: status=%d body=%s, want a conflict naming the reuse", response.Code, response.Body.String())
	}
	for _, bad := range []string{"has space", strings.Repeat("x", maxRequestIDLength+1), "semi;colon"} {
		if code, _ := post(bad, "hello"); code != http.StatusBadRequest {
			t.Errorf("request id %q answered %d, want a bad request", bad, code)
		}
	}
	if got := executions(); got != 1 {
		t.Fatalf("%d executions after refusals, want still one", got)
	}

	// With no number a submission is its own, as before.
	_, first := post("", "plain")
	_, second := post("", "plain")
	if first.Operation == second.Operation {
		t.Fatalf("two unnumbered submissions named one operation %v", first.Operation)
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

// TestRunLedgerForgetsAFailureAndTheOldest checks a submission that failed may be
// numbered again, and that the ledger holds a bounded number of finished
// submissions, the newest, and every unfinished one.
//
// A submission that is still being submitted keeps its key however many
// others finish after it: a repeat of it waits on the same entry and starts
// none of its own, and its key under another fingerprint is still another
// submission's. Once it has finished it is forgotten in its turn, the oldest
// of the finished, and the ledger is back within its capacity.
func TestRunLedgerForgetsAFailureAndTheOldest(t *testing.T) {
	t.Parallel()
	var ledger runLedger
	fingerprint := [32]byte{1}
	entry, owner, err := ledger.begin("a", fingerprint)
	if err != nil || !owner {
		t.Fatalf("first claim = (owner %t, %v)", owner, err)
	}
	waiting, owner, err := ledger.begin("a", fingerprint)
	if err != nil || owner || waiting != entry {
		t.Fatalf("second claim = (owner %t, same entry %t, %v), want to wait on the first", owner, waiting == entry, err)
	}
	ledger.finish("a", entry, artifact.ID{}, errors.New("refused"))
	unfinished, owner, err := ledger.begin("a", fingerprint)
	if err != nil || !owner {
		t.Fatalf("after a failed submission, a claim = (owner %t, %v), want to own it again", owner, err)
	}
	// finished submits and finishes so many others, each under its own key.
	finished := func(prefix string, count int) {
		for index := range count {
			other, _, _ := ledger.begin(fmt.Sprintf("%s%d", prefix, index), fingerprint)
			ledger.finish(fmt.Sprintf("%s%d", prefix, index), other, artifact.ID{}, nil)
		}
	}
	finished("n", ledgerCapacity)
	if repeated, owner, err := ledger.begin("a", fingerprint); err != nil || owner || repeated != unfinished {
		t.Fatalf("a repeat of a submission still being submitted, after %d others finished = (owner %t, same entry %t, %v); want it to wait on the first and start none of its own", ledgerCapacity, owner, repeated == unfinished, err)
	}
	if _, _, err := ledger.begin("a", [32]byte{2}); !errors.Is(err, errRequestIDReused) {
		t.Fatalf("the unfinished submission's key under another fingerprint = %v, want it refused as another submission's", err)
	}
	ledger.finish("a", unfinished, artifact.ID{}, nil)
	finished("m", ledgerCapacity)
	if _, owner, _ := ledger.begin("a", fingerprint); !owner || len(ledger.entries) > ledgerCapacity+1 || len(ledger.order) != len(ledger.entries) {
		t.Fatalf("once finished, the oldest submission is owned again %t with %d entries in an order of %d; want it forgotten in its turn and the ledger within its capacity of %d and the one being claimed", owner, len(ledger.entries), len(ledger.order), ledgerCapacity)
	}
}

// TestRunLedgerSurvivesAFailedSubmissionAndAPanic holds the ledger's order to its
// entries and its owner to a release: a failed submission takes no slot, so the
// retry that follows it is not evicted early by the failed one's stale place; and
// a submission that panics finishes as abandoned, so the callers waiting on it are
// released with an error and the next try owns a submission of its own.
func TestRunLedgerSurvivesAFailedSubmissionAndAPanic(t *testing.T) {
	t.Parallel()
	var ledger runLedger
	fingerprint := [32]byte{2}
	first, _, _ := ledger.begin("a", fingerprint)
	ledger.finish("a", first, artifact.ID{}, errors.New("refused"))
	retry, owner, err := ledger.begin("a", fingerprint)
	if err != nil || !owner {
		t.Fatalf("the retry of a failed submission = (owner %t, %v)", owner, err)
	}
	if len(ledger.order) != 1 {
		t.Fatalf("the order holds %d places after a failure and its retry, want 1", len(ledger.order))
	}
	// The ledger fills to its capacity with the retry as its oldest: it is the one evicted last of the others, never
	// before them, and a failed submission among them leaves its slot free.
	for index := range ledgerCapacity - 1 {
		key := fmt.Sprintf("n%d", index)
		other, _, _ := ledger.begin(key, fingerprint)
		if index == 0 {
			ledger.finish(key, other, artifact.ID{}, errors.New("refused"))
			continue
		}
		ledger.finish(key, other, artifact.ID{}, nil)
	}
	if held, owner, _ := ledger.begin("a", fingerprint); owner || held != retry {
		t.Fatalf("the retry was evicted with the ledger at %d of %d submissions", len(ledger.entries), ledgerCapacity)
	}

	waiting, owner, err := ledger.begin("panics", fingerprint)
	if err != nil || !owner {
		t.Fatalf("claim = (owner %t, %v)", owner, err)
	}
	other, owner, _ := ledger.begin("panics", fingerprint)
	if owner || other != waiting {
		t.Fatal("a second caller did not wait on the first")
	}
	released := make(chan error, 1)
	go func() { _, err := waiting.wait(t.Context()); released <- err }()
	func() {
		defer func() {
			if recover() == nil {
				t.Error("the panic was swallowed")
			}
		}()
		_, _ = ledger.run("panics", waiting, func() (artifact.ID, error) { panic("submitting failed") })
	}()
	if err := <-released; !errors.Is(err, errSubmissionAbandoned) {
		t.Fatalf("a caller waiting on a submission that panicked got %v, want it abandoned", err)
	}
	if _, owner, _ := ledger.begin("panics", fingerprint); !owner {
		t.Fatal("the next try after a panic does not own a submission of its own")
	}
}
