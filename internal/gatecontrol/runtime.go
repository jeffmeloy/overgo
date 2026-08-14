package gatecontrol

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

const RetryFile = "bin/gate_cache.json"

// RetryState binds reusable steps to an exact tree and execution environment.
type RetryState struct {
	treeKey     string
	environment string
	steps       map[string]string
}

type retryEnvelope struct {
	TreeKey     string            `json:"tree_key"`
	Environment string            `json:"environment"`
	Steps       map[string]string `json:"steps"`
}

func (store Store) Retry(treeKey, environment string) RetryState {
	empty := RetryState{treeKey: treeKey, environment: environment, steps: map[string]string{}}
	if treeKey == "" || environment == "" {
		return empty
	}
	raw, err := os.ReadFile(filepath.Join(store.root, filepath.FromSlash(RetryFile)))
	if err != nil {
		return empty
	}
	var envelope retryEnvelope
	if strictjson.DecodeBytes(raw, &envelope) != nil {
		return empty
	}
	state := RetryState{treeKey: envelope.TreeKey, environment: envelope.Environment, steps: envelope.Steps}
	if !state.reusable(treeKey, environment) {
		return empty
	}
	return state
}

func (state RetryState) Succeeded(step string) bool {
	return state.steps[step] == string(runrecord.StepSucceeded)
}

func (state *RetryState) MarkSucceeded(step string) {
	if state.steps == nil {
		state.steps = map[string]string{}
	}
	state.steps[step] = string(runrecord.StepSucceeded)
}

func (store Store) SaveRetry(state RetryState) error {
	if !state.reusable(state.treeKey, state.environment) {
		return fmt.Errorf("gate control: invalid retry state")
	}
	envelope := retryEnvelope{TreeKey: state.treeKey, Environment: state.environment, Steps: state.steps}
	return store.writeJSON(RetryFile, envelope, 0o644)
}

func (state RetryState) reusable(treeKey, environment string) bool {
	if treeKey == "" || environment == "" || state.treeKey != treeKey || state.environment != environment || state.steps == nil {
		return false
	}
	for step, outcome := range state.steps {
		if strings.TrimSpace(step) == "" || outcome != string(runrecord.StepSucceeded) {
			return false
		}
	}
	return true
}

// DiscoverEnvironment creates the immutable gate and retry identity.
func DiscoverEnvironment(root string) (runrecord.Environment, error) {
	command := exec.Command("go", "env", "CGO_ENABLED", "GOFLAGS", "GOEXPERIMENT", "GOTOOLCHAIN")
	command.Dir = root
	out, err := command.CombinedOutput()
	if err != nil {
		return runrecord.Environment{}, fmt.Errorf("go env: %w: %s", err, strings.TrimSpace(string(out)))
	}
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	return environmentFromValues(host, runtime.GOOS, runtime.GOARCH, runtime.Version(), string(out))
}

func environmentFromValues(host, goos, arch, version, output string) (runrecord.Environment, error) {
	values := strings.Split(strings.ReplaceAll(output, "\r\n", "\n"), "\n")
	for len(values) < 4 {
		values = append(values, "")
	}
	return runrecord.NewEnvironment(runrecord.Environment{
		Host: host, OS: goos, Arch: arch, Device: "host", Backend: "go",
		Driver: "cgo=" + strings.TrimSpace(values[0]),
		Runtime: fmt.Sprintf("%s;goflags=%s;goexperiment=%s;gotoolchain=%s",
			version, strings.TrimSpace(values[1]), strings.TrimSpace(values[2]), strings.TrimSpace(values[3])),
	})
}
