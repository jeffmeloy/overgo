package main

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/clioptions"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestSharedModelValidationAdmission(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	path := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Qwen2.5-0.5B-f16.gguf")
	runner := testcheck.Must(modelcli.OpenRunner(t.Context(), roots.Store, path, inference.OpenOptions{}))(t)
	defer runner.Close()
	if runner.ModelID().String() != "model:sha256:2164ee6535ced88ef8f705243eecaad686de45521f7a33e84bff5ba7643587e8" {
		t.Fatal("shared-model fixture identity differs")
	}
	const prompt = "The capital of France is"
	options := inference.GenerateOptions{MaxNewTokens: 32, ContinueAfterEOG: true, DeviceGreedy: true}
	generated := 0
	options.OnToken = func(inference.TokenEvent) error { generated++; return nil }
	before, _, err := runner.Generate(t.Context(), prompt, options)
	if err != nil || generated != options.MaxNewTokens {
		t.Fatalf("initial generation: generated=%d: %v", generated, err)
	}
	report := testcheck.Must(testevidence.RunGoTestCommand(t.Context(), processcontrol.Command{
		Path: "go", Dir: root,
		Args: []string{"test", "-json", "./internal/cuda/device", "-run", "^TestCUDAContextSharedAdmission$", "-count=1"},
	}, testevidence.GoTestOptions{DiagnosticBytes: clioptions.DiagnosticTailBytes}))(t)
	testcheck.NoError(t, testevidence.RequireComplete(report))
	if !report.PackagePassed("overgo/internal/cuda/device") {
		t.Fatal("independent CUDA validation supplied no complete package evidence")
	}
	generated = 0
	after, _, err := runner.Generate(t.Context(), prompt, options)
	if err != nil || generated != options.MaxNewTokens || !slices.Equal(before, after) {
		t.Fatalf("generation after independent GPU tests: %v", err)
	}
	testcheck.NoError(t, runner.Close())
	t.Log("one exact registered 0.5B model stayed resident through independent CUDA validation; both 32-token greedy generations matched; no performance or full-quality claim")
}
