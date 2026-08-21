package inference

import (
	"context"
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
	if err != nil || !binding.deviceNative || !binding.hostRecovery || binding.hostReference {
		t.Fatalf("bound hybrid residency = %+v err=%v", binding, err)
	}
}

func fixtureProgram(spec model.Spec, weights model.Weights) modelrecipe.Plan {
	plan, err := fixtureModelPlan(spec, weights)
	if err != nil {
		panic(err)
	}
	return modelrecipe.Plan{Model: plan}
}

func fixtureModelPlan(spec model.Spec, weights model.Weights) (model.ModelPlan, error) {
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		return model.ModelPlan{}, &model.UnsupportedArchitectureError{Architecture: spec.Architecture}
	}
	return model.CompileModelPlanWithProfile(spec, weights, profile)
}

func fixtureLayerPlan(spec model.Spec, layer int) model.LayerPlan {
	plan, err := fixtureModelPlan(spec, model.Weights{})
	if err != nil {
		panic(err)
	}
	result, err := plan.Layer(uint32(layer))
	if err != nil {
		panic(err)
	}
	return result
}

func fixtureRunner(spec model.Spec, weights model.Weights) *Runner {
	program := fixtureProgram(spec, weights)
	return &Runner{preparedModel: preparedModel{
		spec: program.Model.Spec(), weights: weights, program: program,
	}}
}

func attachFixtureProgram(runner *Runner) *Runner {
	runner.program = fixtureProgram(runner.spec, runner.weights)
	runner.spec = runner.program.Model.Spec()
	return runner
}

func bindFixtureSpec(spec model.Spec) model.Spec {
	return fixtureProgram(spec, model.Weights{}).Model.Spec()
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
	return OpenWithProgram(context.Background(), &loaded, options)
}

func openNativeFixtureRunner(path string, options OpenOptions) (*Runner, error) {
	return openFixtureRunnerWithResidency(path, options, recipe.ResidencyDeviceNative)
}

func openF32FixtureRunner(path string, options OpenOptions) (*Runner, error) {
	return openFixtureRunnerWithResidency(path, options, recipe.ResidencyDeviceF32)
}
