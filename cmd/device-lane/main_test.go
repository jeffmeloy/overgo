package main

import (
	"slices"
	"testing"

	"overgo/internal/devicescope"
)

// TestDeviceLaneProgress holds a step's line to its count and its command, said as the step starts.
func TestDeviceLaneProgress(t *testing.T) {
	if line := deviceProgress([]string{"go", "test", "./internal/cuda/..."}, 1, 3); line != "[device] step 2 of 3: go test ./internal/cuda/..." {
		t.Fatalf("progress %q, want the step's count and command", line)
	}
}

func TestDevicePlan(t *testing.T) {
	steps := deviceSteps(devicescope.VerificationPlan{Packages: []string{"./internal/cuda/kernel", "./internal/optimizer"}})
	if len(steps) != 3 || !slices.Equal(steps[0], []string{"go", "run", "./cmd/cuda-smoke"}) {
		t.Fatalf("scoped steps = %v", steps)
	}
	want := []string{"go", "test", "-json", "-p=1", "-timeout=0", "./internal/cuda/kernel", "-count=1"}
	if !slices.Equal(steps[1], want) {
		t.Fatalf("selected packages = %v, want %v", steps[1], want)
	}
	optimizer := []string{"go", "test", "-json", "-p=1", "-timeout=0", "-run", "^(TestDeviceNewtonSchulzMatchesHost|TestDeviceMuonMatrixStepMatchesHost|TestDeviceMuonPlanResidentMatchesShared|TestDeviceMuonStepPlanMatchesHost)$", "./internal/optimizer", "-count=1"}
	if !slices.Equal(steps[2], optimizer) {
		t.Fatalf("optimizer contracts = %v, want %v", steps[2], optimizer)
	}
	full := deviceSteps(devicescope.VerificationPlan{Full: true})
	if len(full) != 6 {
		t.Fatalf("full steps=%v", full)
	}
	asr := []string{"go", "test", "-json", "-p=1", "-timeout=0", "-run", "^(TestASRResidentCUDA)$", "./internal/speechrecognition", "-count=1"}
	if !slices.ContainsFunc(full, func(step []string) bool { return slices.Equal(step, asr) }) {
		t.Fatalf("full lane omits required ASR execution: %v", full)
	}
	for _, step := range full[1:] {
		if !slices.Contains(step, "-p=1") {
			t.Fatalf("device test step is not serialized: %v", step)
		}
		if !slices.Contains(step, "-timeout=0") {
			t.Fatalf("device test step lacks the no-timeout convention: %v", step)
		}
	}
}

func TestDeviceLaneRunsChangedTestPackage(t *testing.T) {
	steps := deviceSteps(devicescope.VerificationPlan{Packages: []string{"./internal/adaptiveparity"}})
	want := []string{
		"go", "test", "-json", "-p=1", "-timeout=0", "./internal/adaptiveparity", "-count=1",
	}
	if len(steps) != 2 || !slices.Equal(steps[1], want) {
		t.Fatalf("selected test step = %v, want %v", steps, want)
	}
}
