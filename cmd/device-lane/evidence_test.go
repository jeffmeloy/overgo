package main

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/automationcheck"
)

func inferenceDeviceStep(t *testing.T, plan automationcheck.DeviceVerificationPlan) []string {
	t.Helper()
	for _, step := range deviceSteps(plan) {
		if !slices.Contains(step, "./internal/inference") {
			continue
		}
		pattern := stepRunPattern(step)
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
	full := inferenceDeviceStep(t, automationcheck.DeviceVerificationPlan{Full: true})
	partial := inferenceDeviceStep(t, automationcheck.DeviceVerificationPlan{Packages: []string{"./internal/inference"}})
	if !slices.Equal(full, partial) {
		t.Fatalf("full and partial contracts differ: %v / %v", full, partial)
	}
}

func TestDeviceDeclaredEvidence(t *testing.T) {
	plan := automationcheck.DeviceVerificationPlan{Packages: []string{"./internal/inference"}}
	step := inferenceDeviceStep(t, plan)
	names := automationcheck.DeviceScopes(plan)[0].Tests
	stream := func(omit, skip string) string {
		var output strings.Builder
		encoder := json.NewEncoder(&output)
		event := func(action, name string) {
			t.Helper()
			if err := encoder.Encode(map[string]string{"Action": action, "Package": "overgo/internal/inference", "Test": name}); err != nil {
				t.Fatal(err)
			}
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
	if err := verifyDeviceContracts(step, stream("", "")); err != nil {
		t.Fatal(err)
	}
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
	// Last -run wins, as in go test. Each side of a measurement split only
	// requires its own selected tests, while the complete step owns both sides.
	correctness, measurement := splitTestStep(step, names[:1])
	if err := verifyDeviceContracts(correctness, stream(names[0], "")); err != nil {
		t.Fatal(err)
	}
	if err := verifyDeviceContracts(measurement, stream(names[1], "")); err != nil {
		t.Fatal(err)
	}
}
