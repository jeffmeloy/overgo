package main

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/plan"
)

// The driver's defaults, which -init writes: a step takes at most three worker runs before it is set aside, and a run
// of the driver at most forty, and the files are the process state's.
const (
	defaultAttempts                = 3
	defaultInvocations             = 40
	defaultPromptFile              = "loop_prompt.md"
	defaultWorkerName              = "claude-master-lead"
	defaultConfigMode  fs.FileMode = 0o600
)

// defaultWorker is Claude Code given the prompt on its standard input, so a long prompt is passed once and meets no
// command-line limit, with no built-in tool and the six the driver serves it alone (workerTools); its permissions are
// not bypassed, and the repository's hooks apply to it as to any session.
var defaultWorker = []string{"claude", "-p", "--permission-mode", "acceptEdits", "--tools", "", "--strict-mcp-config", "--mcp-config", toolServer,
	"--allowedTools", allowedTools()}

// toolPrefix opens the name a session gives each tool of the server named overgo, and toolServer configures that
// server: go run ./cmd/loop -tools, in the checkout the worker starts in.
const (
	toolPrefix = "mcp__overgo__"
	toolServer = `{"mcpServers": {"overgo": {"command": "go", "args": ["run", "./cmd/loop", "-tools"]}}}`
)

// allowedTools names the worker's tools as its session knows them.
func allowedTools() string {
	var names []string
	for _, tool := range workerTools("", workerStep{}) {
		names = append(names, toolPrefix+tool.name)
	}
	return strings.Join(names, " ")
}

// defaultPrompt opens every worker's prompt: what the worker is for, where the written guidance is, and that it never
// stops to ask. It takes no hold by its words (guard.TakeoverDeclaration): a working session's hold sends the worker
// away. The plan's prompt for the dispatched step follows it.
const defaultPrompt = "You are the overgo driver's worker, in the repository's main checkout. Work the one step below until its " +
	"check passes, and end there: the driver lands it through the gate. skill.md is the written " +
	"guidance. A command that refuses says what to do: do it and keep going, and never stop to ask.\n"

// writeDefaultConfig writes the driver's default configuration at path and its worker's prompt beside it, the worker
// named as the lane's worker, and refuses to write over either.
func writeDefaultConfig(path string, said io.Writer) error {
	if path == "" {
		return errors.New("the checkout's process state holds no place for the driver's configuration")
	}
	prompt := filepath.Join(filepath.Dir(path), defaultPromptFile)
	for _, existing := range []string{path, prompt} {
		if _, err := os.Stat(existing); err == nil || !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("%s already stands: -init writes over nothing", existing)
		}
	}
	defaults := config{Worker: defaultWorker, Name: defaultWorkerName, Prompt: defaultPromptFile,
		Config: loop.Config{MaxAttemptsPerStep: defaultAttempts, MaxInvocations: defaultInvocations}}
	// The replays recorded beside the configuration say for which kinds of step the local worker goes first.
	records, err := readReplayRecords(filepath.Join(filepath.Dir(path), replayRecords))
	if err != nil {
		return err
	}
	ladderFromReplays(&defaults, records)
	// The data root and the role the command runs under are the driver's, which a scheduled task starts with neither.
	defaults.Environment = map[string]string{}
	for _, name := range []string{dataroot.Env, plan.AutomationRoleEnvironment} {
		if value := os.Getenv(name); value != "" {
			defaults.Environment[name] = value
		}
	}
	if err := jsonfile.Write(path, defaults, defaultConfigMode); err != nil {
		return err
	}
	if err := os.WriteFile(prompt, []byte(defaultPrompt), defaultConfigMode); err != nil {
		return err
	}
	_, err = fmt.Fprintf(said, "loop: wrote %s and %s; start the driver with go run ./cmd/loop\n", path, prompt)
	return err
}
