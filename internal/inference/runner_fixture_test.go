package inference

import (
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
)

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
	loaded, err := servingtest.ResolveActiveGGUF(path, recipe.PlacementHybrid)
	if err != nil {
		return nil, err
	}
	return OpenWithProgram(&loaded, options)
}
