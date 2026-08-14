package inference

import (
	"testing"

	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testevidence"
)

func requireIntegration(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
}

func TestBindResidency(t *testing.T) {
	binding, err := bindResidency(recipe.ResidencyHybridNative)
	if err != nil || !binding.deviceNative || !binding.allowFallback || binding.hostReference {
		t.Fatalf("bound hybrid residency = %+v err=%v", binding, err)
	}
}

func fixtureProgram(spec model.Spec, weights model.Weights) modelrecipe.Plan {
	plan, err := model.CompileModelPlan(spec, weights)
	if err != nil {
		panic(err)
	}
	return modelrecipe.Plan{Model: plan}
}

func fixtureRunner(spec model.Spec, weights model.Weights) *Runner {
	return &Runner{preparedModel: preparedModel{
		spec: spec, weights: weights, program: fixtureProgram(spec, weights),
	}}
}

func attachFixtureProgram(runner *Runner) *Runner {
	runner.program = fixtureProgram(runner.spec, runner.weights)
	return runner
}

func openFixtureRunner(path string, deviceOrdinal int) (*Runner, error) {
	return openFixtureRunnerWithOptions(path, OpenOptions{DeviceOrdinal: deviceOrdinal})
}

func openFixtureRunnerWithOptions(path string, options OpenOptions) (*Runner, error) {
	return openFixtureRunnerWithResidency(path, options, recipe.ResidencyHybridNative)
}

func openFixtureRunnerWithResidency(
	path string, options OpenOptions, residency recipe.ResidencyPolicy,
) (*Runner, error) {
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, residency,
	)
	if err != nil {
		return nil, err
	}
	return OpenWithProgram(&loaded, options)
}

func openNativeFixtureRunner(path string, options OpenOptions) (*Runner, error) {
	return openFixtureRunnerWithResidency(path, options, recipe.ResidencyDeviceNative)
}

func openF32FixtureRunner(path string, options OpenOptions) (*Runner, error) {
	return openFixtureRunnerWithResidency(path, options, recipe.ResidencyDeviceF32)
}
