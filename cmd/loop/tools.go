package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/protection"
	"overgo/internal/runrecord"
)

// workerTool is one of the tools the driver serves its worker over the Model Context Protocol (go run ./cmd/loop
// -tools). The worker runs with these alone, no shell, no git and no helper agents: on Nov 7 a helper agent started in
// its own copy of the repository rewrote core.hooksPath and stopped the loop for an hour, and the guard's rules exist
// to catch shell mistakes a worker with no shell cannot make. Each tool takes string arguments by the names it lists.
type workerTool struct {
	name, description string
	arguments         []string
	run               func(arguments map[string]string) (string, error)
}

// approvedProbes are the measurements the worker may run by name, each a command the repository keeps here.
var approvedProbes = map[string]string{
	"status": "go run ./cmd/loop -status",
}

// unwritable are the places no worker edit reaches, whatever its step lists, beside what git ignores.
var unwritable = []string{".git", protection.HarnessConfigDirectory}

// workerStep is the step a worker's tools serve: how it is shown and checked, each answering with what it printed, its
// check's command, and the files it may change, none meaning any a landing admits.
type workerStep struct {
	show, check, verify func() (string, error)
	files               func() ([]string, error)
}

// planStep is the plan's dispatched step in the checkout at root, shown and checked through its cmd/plan.
func planStep(root string) workerStep {
	return planStepFrom(root, liveSource(root))
}

// planStepFrom is the worker's step as source gives it, checked by the plan command in root.
func planStepFrom(root string, source stepSource) workerStep {
	verify := func() (string, error) {
		step, err := source.step()
		return step.Verify, err
	}
	files := func() ([]string, error) {
		step, err := source.step()
		return step.Files, err
	}
	check := func() (string, error) { return runTool("go", "-C", root, "run", "./cmd/plan", "-verify") }
	return workerStep{show: source.reasoned, check: check, verify: verify, files: files}
}

// missingTest tells a worker whose check ran no test which test the check names and where it may write it: a real
// step's check often names the test the step adds, and a worker not told so reads and never writes (on 2026-11-09 the
// local worker spent 81 calls on clip-encode/admit-runtime and edited nothing).
func missingTest(step workerStep) string {
	command, err := step.verify()
	parsed, parseErr := runrecord.ParseVerify(command)
	files, filesErr := step.files()
	if errors.Join(err, parseErr, filesErr) != nil {
		return ""
	}
	var named []string
	for _, test := range parsed.GoTests() {
		if test.Target != nil {
			named = append(named, strings.Trim(test.Target.Pattern, "^$")+" in "+strings.Join(test.Packages, " "))
		}
	}
	return fmt.Sprintf("\nThe check ran no test: the test it names, %s, is in no file yet. Write it first, in a file the step may change (%s): it fails before the change and passes after it. Then make the change.",
		strings.Join(named, "; "), strings.Join(files, ", "))
}

// workerTools are the nine tools over the checkout at root, serving step.
func workerTools(root string, step workerStep) []workerTool {
	inside := func(name string) (string, error) {
		relative := filepath.ToSlash(filepath.Clean(name))
		if filepath.IsAbs(name) || relative == ".." || strings.HasPrefix(relative, "../") {
			return "", fmt.Errorf("%s is outside the checkout", name)
		}
		return relative, nil
	}
	shown := func(run func() (string, error)) string {
		text, err := run()
		if err != nil {
			return text + "\nfailed: " + err.Error()
		}
		return text
	}
	tool := func(arguments ...string) string {
		return shown(func() (string, error) { return runTool(arguments...) })
	}
	// writable is a path the worker may change: inside the checkout, a place git tracks, none the guard holds, and
	// among the step's files.
	writable := func(name string) (string, error) {
		relative, err := inside(name)
		if err != nil {
			return "", err
		}
		files, err := step.files()
		if err != nil {
			return "", err
		}
		// What git ignores is no worker's to write: the reference tree, the models and scratch among it.
		_, tracked := gitauthority.Query(context.Background(), root, "check-ignore", "-q", "--", relative)
		if plan.Lists(unwritable, relative) || tracked == nil || len(files) != 0 && !plan.Lists(files, relative) {
			return "", fmt.Errorf("%s is not among the files the step may change (%s)", relative, strings.Join(files, ", "))
		}
		return relative, nil
	}
	return []workerTool{
		{"read_file", "Read a file of the checkout, by its path from the checkout's root.", []string{"path"}, func(arguments map[string]string) (string, error) {
			relative, err := inside(arguments["path"])
			if err != nil {
				return "", err
			}
			data, err := os.ReadFile(filepath.Join(root, relative))
			return string(data), err
		}},
		// A worker that can only read whole files reads its context full before it edits anything: on 2026-11-08
		// Qwen3.8-27B spent 39 of its 40 replies on file reads and changed nothing (local-worker/replay).
		{"read_lines", "Read lines from to to of a file of the checkout, by its path from the checkout's root, each with its number.", []string{"path", "from", "to"}, func(arguments map[string]string) (string, error) {
			relative, err := inside(arguments["path"])
			if err != nil {
				return "", err
			}
			from, fromErr := strconv.Atoi(arguments["from"])
			to, toErr := strconv.Atoi(arguments["to"])
			if fromErr != nil || toErr != nil || from < 1 || to < from {
				return "", errors.New("from and to are line numbers, from 1, to no less than from")
			}
			data, err := os.ReadFile(filepath.Join(root, relative))
			if err != nil {
				return "", err
			}
			var text strings.Builder
			number := 0
			for line := range strings.Lines(string(data)) {
				if number++; number >= from && number <= to {
					fmt.Fprintf(&text, "%d\t%s", number, line)
				}
			}
			return text.String(), nil
		}},
		// The text is found as written within one line, which the worker took for a pattern and a span of lines ("func
		// Test.*Store", "# overgo\n") in runs 3 to 7 of loop-proof/local-demonstrated, reading each miss as the code's
		// absence (loop-proof/search-literal).
		{"search_files", "Find a text in the checkout's files, those git does not track yet among them: each line holding it, with its file and line number. The text is found as written, within one line, not as a pattern.", []string{"text"}, func(arguments map[string]string) (string, error) {
			if arguments["text"] == "" {
				return "", errors.New("the text to find is empty")
			}
			// A file git does not track yet is searched too, short of what it ignores: a replay puts a landing's added
			// test at its parent untracked, and in run 6 of loop-proof/local-demonstrated every failed step's check was
			// such a file, out of the worker's sight (loop-proof/search-untracked).
			found, err := gitauthority.Query(context.Background(), root, "grep", "--untracked", "-n", "-I", "-F", "-e", arguments["text"])
			if err != nil && len(found) == 0 {
				return "no line of the checkout's files holds this text as written (the search takes no pattern)", nil
			}
			return string(found), nil
		}},
		{"edit_file", "Replace the one occurrence of old with new in a file the step may change; an empty old writes new as the whole file, creating the file and its folders where they do not exist.", []string{"path", "old", "new"}, func(arguments map[string]string) (string, error) {
			relative, err := writable(arguments["path"])
			if err != nil {
				return "", err
			}
			path := filepath.Join(root, relative)
			text := arguments["new"]
			if old := arguments["old"]; old != "" {
				data, err := os.ReadFile(path)
				if err != nil {
					return "", err
				}
				if count := strings.Count(string(data), old); count != 1 {
					return "", fmt.Errorf("old occurs %d times in %s; give text that occurs once", count, relative)
				}
				text = strings.Replace(string(data), old, text, 1)
			}
			if err := os.MkdirAll(filepath.Dir(path), clioptions.OutputDirectoryMode); err != nil {
				return "", err
			}
			// A file without its final newline prints as one with it, so the edit says so: in loop-proof/
			// local-demonstrated the worker wrote CLAUDE.md as "@AGENTS.md" with none, its check wanting one, and searched
			// for a line-ending setting it could not see (run 7).
			said := "edited " + relative
			if text != "" && !strings.HasSuffix(text, "\n") {
				said += ", which now ends without a final newline"
			}
			return said, os.WriteFile(path, []byte(text), clioptions.OutputFileMode)
		}},
		// A step whose change is a deletion needs one: test-reach/fuzz-or-delete's check wants a script gone, and with no
		// way to remove a file the worker spent its 40 replies three times (loop-proof/local-demonstrated, run 5).
		{"delete_file", "Delete a file the step may change.", []string{"path"}, func(arguments map[string]string) (string, error) {
			relative, err := writable(arguments["path"])
			if err != nil {
				return "", err
			}
			return "deleted " + relative, os.Remove(filepath.Join(root, relative))
		}},
		{"run_check", "Run the step's check (go run ./cmd/plan -verify).", nil, func(map[string]string) (string, error) {
			said := shown(step.check)
			if step.verify != nil && strings.Contains(said, planverify.ErrVacuous.Error()) {
				said += missingTest(step)
			}
			return said, nil
		}},
		{"run_tests", "Run one package's tests, by its directory from the checkout's root.", []string{"package"}, func(arguments map[string]string) (string, error) {
			relative, err := inside(arguments["package"])
			if err != nil || strings.HasPrefix(relative, "-") {
				return "", errors.Join(err, errors.New("the package is a directory of the checkout"))
			}
			return tool("go", "-C", root, "test", "./"+relative, "-count=1"), nil
		}},
		{"show_step", "Show the step: its title, check, notes and reasoning (go run ./cmd/plan -prompt).", nil, func(map[string]string) (string, error) {
			return shown(step.show), nil
		}},
		{"run_probe", "Run an approved probe by its name.", []string{"name"}, func(arguments map[string]string) (string, error) {
			command, approved := approvedProbes[arguments["name"]]
			if !approved {
				return "", fmt.Errorf("%q is no approved probe", arguments["name"])
			}
			return tool(strings.Fields(command)...), nil
		}},
	}
}

// serveTools answers the worker's requests on in with tools, one JSON-RPC message a line, until in ends: the protocol's
// opening, the tool list and calls. A call that fails answers as an error the worker reads, and the server goes on.
func serveTools(in io.Reader, out io.Writer, tools []workerTool) error {
	decoder, encoder := json.NewDecoder(in), json.NewEncoder(out)
	for {
		var request struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params struct {
				ProtocolVersion string         `json:"protocolVersion"`
				Name            string         `json:"name"`
				Arguments       map[string]any `json:"arguments"`
			} `json:"params"`
		}
		if err := decoder.Decode(&request); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return err
		}
		// A notification asks no answer.
		if request.ID == nil {
			continue
		}
		var result any = map[string]any{}
		switch request.Method {
		case "initialize":
			result = map[string]any{"protocolVersion": request.Params.ProtocolVersion, "capabilities": map[string]any{"tools": map[string]any{}},
				"serverInfo": map[string]string{"name": "overgo", "version": "1"}}
		case "tools/list":
			listing := make([]map[string]any, len(tools))
			for index, tool := range tools {
				listing[index] = map[string]any{"name": tool.name, "description": tool.description, "inputSchema": tool.schema()}
			}
			result = map[string]any{"tools": listing}
		case "tools/call":
			text, err := callTool(tools, request.Params.Name, request.Params.Arguments)
			if err != nil {
				text = err.Error()
			}
			result = map[string]any{"content": []map[string]string{{"type": "text", "text": text}}, "isError": err != nil}
		}
		if err := encoder.Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": result}); err != nil {
			return err
		}
	}
}

// schema is the JSON schema of the tool's arguments: an object of the strings it names, each required.
func (tool workerTool) schema() map[string]any {
	properties := map[string]any{}
	for _, argument := range tool.arguments {
		properties[argument] = map[string]string{"type": "string"}
	}
	return map[string]any{"type": "object", "properties": properties, "required": append([]string{}, tool.arguments...)}
}

// callTool runs the tool named with the arguments given, each it names a string the call gives: a call missing one is
// refused before the tool runs, since an edit read a missing old as empty and wrote new over the whole file (SQA-347).
// Arguments it does not name are not read.
func callTool(tools []workerTool, name string, given map[string]any) (string, error) {
	at := slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == name })
	if at < 0 {
		return "", fmt.Errorf("%q is no tool of the worker's", name)
	}
	arguments := map[string]string{}
	for _, argument := range tools[at].arguments {
		text, isText := given[argument].(string)
		if !isText {
			return "", fmt.Errorf("%s takes %s, a string, which the call did not give (%v)", name, argument, given[argument])
		}
		arguments[argument] = text
	}
	return tools[at].run(arguments)
}
