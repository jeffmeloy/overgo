package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// toolAnswer is the part of the tool server's answer the test reads.
type toolAnswer struct {
	Result struct {
		ProtocolVersion string                  `json:"protocolVersion"`
		Tools           []struct{ Name string } `json:"tools"`
		IsError         bool                    `json:"isError"`
	} `json:"result"`
}

// TestTheWorkerHasOnlyItsTools holds the worker to the eight tools the driver serves it: on Nov 7 a helper agent with a
// shell rewrote core.hooksPath and stopped the loop for an hour. The server answers the protocol's opening, lists
// exactly the eight, refuses an edit outside the step's files and makes one inside them; the default worker is given
// no built-in tool and is allowed these eight alone.
func TestTheWorkerHasOnlyItsTools(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	tools := workerTools(root, workerStep{files: func() ([]string, error) { return []string{"cmd/x"}, nil }})
	requests := strings.Join([]string{
		`{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}`,
		`{"jsonrpc": "2.0", "method": "notifications/initialized"}`,
		`{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}`,
		`{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "internal/y/y.go", "old": "", "new": "package y\n"}}}`,
		`{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "cmd/x/x.go", "old": "", "new": "package main\n"}}}`,
	}, "\n")
	var out bytes.Buffer
	testcheck.NoError(t, serveTools(strings.NewReader(requests), &out, tools))
	var answers []toolAnswer
	for line := range strings.Lines(out.String()) {
		var answer toolAnswer
		testcheck.NoError(t, json.Unmarshal([]byte(line), &answer))
		answers = append(answers, answer)
	}
	if len(answers) != 4 || answers[0].Result.ProtocolVersion != "2025-06-18" {
		t.Fatalf("the server answered %d requests, opening with %+v; want the four with ids, the protocol echoed", len(answers), answers)
	}
	var listed []string
	for _, tool := range answers[1].Result.Tools {
		listed = append(listed, tool.Name)
	}
	if want := []string{"read_file", "read_lines", "search_files", "edit_file", "run_check", "run_tests", "show_step", "run_probe"}; !slices.Equal(listed, want) {
		t.Fatalf("the worker's tools are %v, want %v", listed, want)
	}
	if _, err := os.Stat(filepath.Join(root, "internal", "y", "y.go")); !answers[2].Result.IsError || err == nil {
		t.Fatalf("an edit outside the step's files answered error %v and wrote the file (%v); want it refused", answers[2].Result.IsError, err)
	}
	if data, err := os.ReadFile(filepath.Join(root, "cmd", "x", "x.go")); answers[3].Result.IsError || err != nil || string(data) != "package main\n" {
		t.Fatalf("an edit inside the step's files answered error %v and left %q (%v)", answers[3].Result.IsError, data, err)
	}
	allowed := make([]string, len(listed))
	for index, name := range listed {
		allowed[index] = toolPrefix + name
	}
	at := slices.Index(defaultWorker, "--allowedTools")
	if builtins := slices.Index(defaultWorker, "--tools"); at < 0 || builtins < 0 || defaultWorker[builtins+1] != "" || defaultWorker[at+1] != strings.Join(allowed, " ") {
		t.Fatalf("the default worker is %q, want no built-in tool and the eight alone allowed", defaultWorker)
	}
}
