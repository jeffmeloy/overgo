package trainingworkflow

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// chainChildEnvironment carries the chain probe's request to a child process the test starts as itself.
const chainChildEnvironment = "OVERGO_CHAIN_REPLAY_KEY_CHILD"

// chainChildRequest is one chain probe a child process runs.
type chainChildRequest struct {
	Store   string `json:"store"`
	Scorer  string `json:"scorer"`
	Drafter string `json:"drafter"`
}

// TestAReplayKeyNamesTheCodeThatComputesIt holds the chain probe's replay keys to the code that drafts and tokenizes.
//   - one request = one store, one scorer, one drafter, one protocol, one window
//   - a process of a test's build names only itself as its code, so two processes are two codes
//   - a repeat by other code => every stage of both arms runs again
//   - a repeat answered from the first process's stage receipts is refused, because code that is gone gave those answers
func TestAReplayKeyNamesTheCodeThatComputesIt(t *testing.T) {
	if encoded := os.Getenv(chainChildEnvironment); encoded != "" {
		// Serial: it is the one test of a child process, which runs the probe the environment names.
		runChainChild(t, encoded)
		return
	}
	t.Parallel()
	t.Run("chain", func(t *testing.T) {
		directory := t.TempDir()
		request := chainChildRequest{
			Store:  filepath.Join(directory, "store"),
			Scorer: filepath.Join(directory, "scorer"), Drafter: filepath.Join(directory, "drafter"),
		}
		for seed, model := range map[int64]string{6: request.Scorer, 7: request.Drafter} {
			weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
				Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: seed,
			})
			writeModel(t, model, weights, shapes)
		}
		encoded, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		first := 0
		for process := 1; process <= 2; process++ {
			command := exec.CommandContext(t.Context(), executable, "-test.run=^TestAReplayKeyNamesTheCodeThatComputesIt$")
			command.Env = append(os.Environ(), chainChildEnvironment+"="+string(encoded))
			output, exit := command.CombinedOutput()
			ran := completedChainStages(t, request.Store)
			if process == 1 {
				first = ran
			}
			if exit != nil || first == 0 || ran != process*first {
				t.Fatalf("process %d ran the chain probe with one request and ended with %v, and the store then held %d completed stages, want no error and %d stages, more than none: a repeat by other code was taken from the stage receipts the first process left: %s", process, exit, ran, process*first, output)
			}
		}
	})
}

// runChainChild runs the chain probe a parent test asked this process for.
func runChainChild(t *testing.T, encoded string) {
	t.Helper()
	var request chainChildRequest
	if err := json.Unmarshal([]byte(encoded), &request); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(request.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result, err := RunChainViability(store, ChainConfig{
		ScorerDir: request.Scorer, DrafterDir: request.Drafter,
		Prefix: chainMinimumTokens, Draft: 1, Target: chainMinimumTokens,
		Windows: [][]int{{1, 2, 3, 4, 1}},
	})
	if err != nil || len(result.Outcomes) != 1 {
		t.Fatalf("the chain probe gave %d outcomes (%v), want one", len(result.Outcomes), err)
	}
}

// completedChainStages counts the stage receipts of the store at repository that record a completed stage.
//   - a stage that runs to its end publishes one such receipt
//   - a stage answered from an earlier receipt publishes none
func completedChainStages(t *testing.T, repository string) (completed int) {
	t.Helper()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	receipts := overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.StageReceiptMediaType, Schema: runrecord.StageReceiptSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}
	if _, err := store.VisitDocuments(t.Context(), receipts, func(view overgodb.DocumentView) error {
		receipt, err := runrecord.ParseStageReceipt(view.Content.Data)
		if err == nil && receipt.State == runrecord.StageCompleted {
			completed++
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return completed
}
