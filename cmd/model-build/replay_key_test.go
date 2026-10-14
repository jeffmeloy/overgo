package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
	"overgo/internal/workflowruntime"
)

// commandEnvironment marks a child process a test started as the model-build command.
const commandEnvironment = "OVERGO_MODEL_BUILD_TEST_COMMAND"

// TestMain runs the tests, or the command's own main in a child a test started as the command.
func TestMain(m *testing.M) {
	if os.Getenv(commandEnvironment) == "" {
		os.Exit(m.Run())
	}
	main()
	// main returns only when the command succeeded.
	os.Exit(0)
}

// TestAReplayKeyNamesTheCodeThatComputesIt holds the command's replay key to the code that builds its model.
//   - one request = one store, one corpus, one seed, one step count
//   - a process of a test's build names only itself as its code, so two processes are two codes
//   - a repeat by other code => every stage of the build runs again
//   - a repeat answered from the first process's stage receipts is refused, because code that is gone gave those answers
func TestAReplayKeyNamesTheCodeThatComputesIt(t *testing.T) {
	t.Parallel()
	t.Run("model-build", func(t *testing.T) {
		directory := t.TempDir()
		dataset := filepath.Join(directory, "corpus.json")
		if err := os.WriteFile(dataset, []byte(`["abcd","bcda","cdab","dabc"]`), 0o600); err != nil {
			t.Fatal(err)
		}
		repository := filepath.Join(directory, "store")
		store, err := overgodb.Open(repository)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := scratchmodel.PublishDerivationProfileCatalog(t.Context(), store); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		stages := len(workflowruntime.ModelBuildStages())
		for process := 1; process <= 2; process++ {
			command := exec.CommandContext(t.Context(), executable, "-repo", repository, "-dataset", dataset, "-steps", "1")
			// The child's refusal is this test's to read, and goes to no error file a caller of the tests named.
			command.Env = append(os.Environ(), commandEnvironment+"=1", clioptions.ErrorFileEnvironment+"=")
			if output, err := command.CombinedOutput(); err != nil {
				t.Fatalf("process %d of the command: %v: %s", process, err, output)
			}
			if ran := completedStages(t, repository); ran != process*stages {
				t.Fatalf("after process %d ran the command with one corpus, seed and step count, the store held %d completed stages, want %d: a repeat by other code was answered from the stage receipts the first process left", process, ran, process*stages)
			}
		}
	})
}

// completedStages counts the stage receipts of the store at repository that record a completed stage.
//   - a stage that runs to its end publishes one such receipt
//   - a stage answered from an earlier receipt publishes none
func completedStages(t *testing.T, repository string) (completed int) {
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
