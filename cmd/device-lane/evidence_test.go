package main

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/devicescope"
	"overgo/internal/testcheck"
)

func inferenceDeviceStep(t *testing.T, plan devicescope.VerificationPlan) []string {
	t.Helper()
	for _, step := range deviceSteps(plan) {
		if !slices.Contains(step, "./internal/inference") {
			continue
		}
		pattern := stepTestPattern(step, "-run")
		if pattern == nil || !pattern.MatchString("TestRecurrentCheckpoint") || !pattern.MatchString("TestNextNMTPDeviceSpeculationLossless") {
			t.Fatalf("inference step omits required contracts: %v", step)
		}
		for _, host := range []string{"TestBindResidency", "TestContinuationScoringPathsAgreeOnAModel", "TestRecurrentCheckpointRenamed"} {
			if pattern.MatchString(host) {
				t.Fatalf("device selection includes unrelated test %s: %v", host, step)
			}
		}
		return step
	}
	t.Fatal("no inference device command")
	return nil
}

func TestDeviceDeclaredCommands(t *testing.T) {
	full := inferenceDeviceStep(t, devicescope.VerificationPlan{Full: true})
	partial := inferenceDeviceStep(t, devicescope.VerificationPlan{Packages: []string{"./internal/inference"}})
	if !slices.Equal(full, partial) {
		t.Fatalf("full and partial contracts differ: %v / %v", full, partial)
	}
}

func TestDeviceDeclaredEvidence(t *testing.T) {
	plan := devicescope.VerificationPlan{Packages: []string{"./internal/inference"}}
	step := inferenceDeviceStep(t, plan)
	names := devicescope.Scopes(plan)[0].Tests
	stream := func(omit, skip string) string {
		var output strings.Builder
		encoder := json.NewEncoder(&output)
		event := func(action, name string) {
			t.Helper()
			testcheck.NoError(t, encoder.Encode(map[string]string{"Action": action, "Package": "overgo/internal/inference", "Test": name}))
		}
		event("start", "")
		for _, name := range names {
			if name == omit {
				continue
			}
			event("run", name)
			if name == skip {
				event("skip", name)
			} else {
				event("pass", name)
			}
		}
		event("pass", "")
		return output.String()
	}
	testcheck.NoError(t, verifyDeviceContracts(step, stream("", "")))
	for _, name := range names {
		if err := verifyDeviceContracts(step, stream(name, "")); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("missing %s accepted: %v", name, err)
		}
		if err := verifyDeviceContracts(step, stream("", name)); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("skipped %s accepted: %v", name, err)
		}
	}
	if err := verifyDeviceContracts(step, ""); err == nil {
		t.Fatal("empty evidence accepted")
	}
	if err := verifyDeviceContracts([]string{"cuda-smoke.exe"}, ""); err != nil {
		t.Fatalf("probe without declared test scope: %v", err)
	}
	// Last -run wins, as in go test: a step that selects some tests requires only those, and -skip passes over its
	// own.
	testcheck.NoError(t, verifyDeviceContracts(append(slices.Clone(step), "-run", "^"+names[1]+"$"), stream(names[0], "")))
	testcheck.NoError(t, verifyDeviceContracts(append(slices.Clone(step), "-skip", "^"+names[1]+"$"), stream(names[1], "")))
}
