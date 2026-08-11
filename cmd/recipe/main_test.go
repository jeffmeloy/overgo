package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

const forecastInputFixture = `[1.25,-2.5,4]`

var forecastValuesFixture = []float32{1.25, -2.5, 4}

func TestForecastCommandResolvesExecutableRecipeStage(t *testing.T) {
	capability, ok := capabilityCommands[recipe.TaskForecast]
	if !ok || capability.inventory == nil || capability.definition == nil || capability.execute == nil {
		t.Fatalf("forecast capability = %+v", capability)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "forecast-command-model")
	definition, err := capability.definition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	stages := program.Stages()
	if len(stages) != 1 || stages[0].Module.ID != modelrecipe.ModuleForecastSeries {
		t.Fatalf("forecast program = %+v", program)
	}
}

func TestForecastInputIsCanonicalArtifact(t *testing.T) {
	values, content, err := forecastInput(forecastInputFixture)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != len(forecastValuesFixture) {
		t.Fatalf("values = %v", values)
	}
	for index, want := range forecastValuesFixture {
		if values[index] != want {
			t.Fatalf("values[%d] = %g, want %g", index, values[index], want)
		}
	}
	if err := content.Validate(); err != nil || content.Descriptor.ID.Kind() != artifact.KindFile {
		t.Fatalf("input content = (%+v, %v)", content.Descriptor, err)
	}
}
