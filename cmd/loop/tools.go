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
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/protection"
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

// workerTools are the six tools over the checkout at root; listed gives the files the dispatched step may change, none
// meaning any a landing admits.
func workerTools(root string, listed func() ([]string, error)) []workerTool {
	inside := func(name string) (string, error) {
		relative := filepath.ToSlash(filepath.Clean(name))
		if filepath.IsAbs(name) || relative == ".." || strings.HasPrefix(relative, "../") {
			return "", fmt.Errorf("%s is outside the checkout", name)
		}
		return relative, nil
	}
	tool := func(arguments ...string) string {
		shown, err := runTool(arguments...)
		if err != nil {
			return shown + "\nfailed: " + err.Error()
		}
		return shown
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
		{"edit_file", "Replace the one occurrence of old with new in a file the step may change; an empty old writes new as the whole file.", []string{"path", "old", "new"}, func(arguments map[string]string) (string, error) {
			relative, err := inside(arguments["path"])
			if err != nil {
				return "", err
			}
			files, err := listed()
			if err != nil {
				return "", err
			}
			// What git ignores is no worker's to write: the reference tree, the models and scratch among it.
			_, tracked := gitauthority.Query(context.Background(), root, "check-ignore", "-q", "--", relative)
			if plan.Lists(unwritable, relative) || tracked == nil || len(files) != 0 && !plan.Lists(files, relative) {
				return "", fmt.Errorf("%s is not among the files the step may change (%s)", relative, strings.Join(files, ", "))
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
			return "edited " + relative, os.WriteFile(path, []byte(text), clioptions.OutputFileMode)
		}},
		{"run_check", "Run the step's check (go run ./cmd/plan -verify).", nil, func(map[string]string) (string, error) {
			return tool("go", "run", "./cmd/plan", "-verify"), nil
		}},
		{"run_tests", "Run one package's tests, by its directory from the checkout's root.", []string{"package"}, func(arguments map[string]string) (string, error) {
			relative, err := inside(arguments["package"])
			if err != nil || strings.HasPrefix(relative, "-") {
				return "", errors.Join(err, errors.New("the package is a directory of the checkout"))
			}
			return tool("go", "test", "./"+relative, "-count=1"), nil
		}},
		{"show_step", "Show the step: its title, check, notes and reasoning (go run ./cmd/plan -prompt).", nil, func(map[string]string) (string, error) {
			return tool("go", "run", "./cmd/plan", "-prompt"), nil
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
				properties := map[string]any{}
				for _, argument := range tool.arguments {
					properties[argument] = map[string]string{"type": "string"}
				}
				listing[index] = map[string]any{"name": tool.name, "description": tool.description,
					"inputSchema": map[string]any{"type": "object", "properties": properties, "required": tool.arguments}}
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

// callTool runs the tool named with the arguments given, each read as a string.
func callTool(tools []workerTool, name string, given map[string]any) (string, error) {
	at := slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == name })
	if at < 0 {
		return "", fmt.Errorf("%q is no tool of the worker's", name)
	}
	arguments := map[string]string{}
	for key, value := range given {
		arguments[key] = fmt.Sprint(value)
	}
	return tools[at].run(arguments)
}
