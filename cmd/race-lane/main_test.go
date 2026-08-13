package main

import (
	"slices"
	"strings"
	"testing"
)

func TestHostRaceCmd(t *testing.T) {
	cmd := hostRaceCmd()
	if got := strings.Join(cmd, " "); !strings.HasPrefix(got, "go test -race -count=1 ") {
		t.Fatalf("host race command not a race-enabled uncached go test: %q", got)
	}
	if !slices.Contains(cmd, "-race") {
		t.Fatal("host race command must pass -race")
	}
	for _, pkg := range hostRacePkgs {
		if !slices.Contains(cmd, pkg) {
			t.Errorf("host race command missing package %q", pkg)
		}
	}
	// CGO must not be an argument; it is an environment fact, set at exec.
	if slices.ContainsFunc(cmd, func(s string) bool { return strings.Contains(s, "CGO") }) {
		t.Error("CGO belongs in the environment, not the command line")
	}
}

func TestSanitizerCmd(t *testing.T) {
	cmd := sanitizerCmd("racecheck", "/tmp/x.test", "-test.run", "Device")
	want := []string{sanitizerTool, "--tool", "racecheck", "--error-exitcode", "1", "/tmp/x.test", "-test.run", "Device"}
	if !slices.Equal(cmd, want) {
		t.Fatalf("sanitizer command\n got %v\nwant %v", cmd, want)
	}
}

func TestSanitizerCmdErrorExitcode(t *testing.T) {
	// Without --error-exitcode a detected hazard would exit 0 and pass silently.
	for _, tool := range deviceRaceTools {
		cmd := sanitizerCmd(tool, "bin")
		i := slices.Index(cmd, "--error-exitcode")
		if i < 0 || i+1 >= len(cmd) || cmd[i+1] != "1" {
			t.Errorf("%s must set --error-exitcode 1 so a hazard is a FAIL: %v", tool, cmd)
		}
	}
}

func TestDeviceRaceToolsAreRaceTools(t *testing.T) {
	// The race lane runs the two hazard detectors, not the memory-only tools.
	if !slices.Contains(deviceRaceTools, "racecheck") || !slices.Contains(deviceRaceTools, "synccheck") {
		t.Fatalf("device race tools must include racecheck and synccheck: %v", deviceRaceTools)
	}
}

func TestVerdict(t *testing.T) {
	if verdict(nil) != "ok" {
		t.Error("nil error must read ok")
	}
	if verdict(errFail{}) != "FAIL" {
		t.Error("non-nil error must read FAIL")
	}
}

func TestRanLabel(t *testing.T) {
	if ranLabel(true) != "ran" {
		t.Error("ran lane must report ran")
	}
	if !strings.Contains(ranLabel(false), "SKIPPED") {
		t.Error("unrun lane must report SKIPPED, never silence")
	}
}

type errFail struct{}

func (errFail) Error() string { return "fail" }
