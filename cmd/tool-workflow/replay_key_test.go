package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync/atomic"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// commandEnvironment marks a child process a test started as the tool-workflow command.
const commandEnvironment = "OVERGO_TOOL_WORKFLOW_TEST_COMMAND"

// TestMain runs the tests, or the command's own main in a child a test started as the command.
//   - child's flags = the command's alone, because the test binary's flags are not the command's
func TestMain(m *testing.M) {
	if os.Getenv(commandEnvironment) == "" {
		os.Exit(m.Run())
	}
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	main()
	// main returns only when the command succeeded.
	os.Exit(0)
}

// TestAReplayKeyNamesTheCodeThatComputesIt holds the command's replay key to the code that runs its tools.
//   - one request = one store, one key, one request file
//   - a process of a test's build names only itself as its code, so two processes are two codes
//   - a repeat by other code => the tool runs again
//   - a repeat answered from the first process's stage receipt is refused, because code that is gone gave that answer
func TestAReplayKeyNamesTheCodeThatComputesIt(t *testing.T) {
	t.Parallel()
	t.Run("tool-workflow", func(t *testing.T) {
		var calls atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
			calls.Add(1)
			response.Header().Set("Content-Type", "application/json")
			_, _ = response.Write([]byte(`{"ok":true}`))
		}))
		defer server.Close()
		manual := testcheck.Must(agenttool.NewManual(agenttool.Manual{
			Name: "state.inspect", Description: "Replay key test tool.", Effect: agenttool.EffectInspection,
			Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/inspect"},
		}))(t)
		model := testutil.ArtifactID(t, artifact.KindModel, "replay-key-tool-workflow-model")
		directory := t.TempDir()
		repository := filepath.Join(directory, "store")
		store := testcheck.Must(overgodb.Create(repository))(t)
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "replay-key/model", Artifacts: []artifact.Descriptor{{ID: model}}}); err != nil {
			t.Fatal(err)
		}
		if _, err := agenttool.PublishManualCatalog(t.Context(), store, []agenttool.Manual{manual}); err != nil {
			t.Fatal(err)
		}
		testcheck.NoError(t, store.Close())
		request := testcheck.Must(json.Marshal(workflowRequest{
			Proposal:  agenttool.ToolWorkflowProposal{Model: model, Steps: []agenttool.ToolWorkflowStep{{ID: "inspect", Manual: manual.ID}}},
			Arguments: map[recipe.NodeID]json.RawMessage{"inspect": json.RawMessage(`{}`)},
		}))(t)
		requestPath := filepath.Join(directory, "request.json")
		testcheck.NoError(t, os.WriteFile(requestPath, request, 0o600))
		executable := testcheck.Must(os.Executable())(t)
		for process := int32(1); process <= 2; process++ {
			command := exec.CommandContext(t.Context(), executable, "-key", "replay-key", "-repo", repository, requestPath)
			// The child's refusal is this test's to read, and goes to no error file a caller of the tests named.
			command.Env = append(os.Environ(), commandEnvironment+"=1", clioptions.ErrorFileEnvironment+"=")
			output, exit := command.CombinedOutput()
			if exit != nil {
				t.Fatalf("workflow command failed: %v: %s", exit, output)
			}
			var result struct {
				Outputs map[recipe.PortName]workflowruntime.Value
				Run     json.RawMessage
				RunID   artifact.ID
				Commit  artifact.CommitID
			}
			if err := json.Unmarshal(output, &result); err != nil {
				t.Fatalf("workflow result: %v: %s", err, output)
			}
			var canonical bytes.Buffer
			testcheck.NoError(t, json.Compact(&canonical, result.Run))
			run, err := runrecord.ParseRun(canonical.Bytes())
			if err != nil || run.ID != result.RunID || process == 1 && !result.Commit.Valid() || len(result.Outputs) == 0 {
				t.Fatalf("workflow identities or outputs: %+v, %v", result, err)
			}
			if ran := calls.Load(); ran != process {
				t.Fatalf("after process %d ran the command with one key and one request, the tool had run %d times, want %d: a repeat by other code was answered from the stage receipt the first process left (the command ended with %v: %s)", process, ran, process, exit, output)
			}
		}
	})
}
