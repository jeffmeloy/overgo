package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/tabularicl"
	"overgo/internal/testutil"
)

const (
	forecastInputFixture       = `[1.25,-2.5,4]`
	tabularInputFixture        = `{"task":"classification","x":[2,3],"y":[1,0],"rows":2,"cols":1,"train_rows":1}`
	invalidTabularInputFixture = `{"task":"classification","unknown":1}`
	seq2seqInputFixture        = `{"source":[3,5],"max_tokens":2}`
	speechInputFixture         = `{"text":"hello","max_frames":2,"seed":7}`
	tabularInputRows           = 2
	tabularInputCols           = 1
	tabularInputTrainRows      = 1
)

var forecastValuesFixture = []float32{1.25, -2.5, 4}

func TestCommandsResolveExecutableCapabilityStages(t *testing.T) {
	tests := []struct {
		task    recipe.Task
		modules []recipe.ModuleID
	}{
		{task: recipe.TaskForecast, modules: []recipe.ModuleID{modelrecipe.ModuleForecastSeries}},
		{task: recipe.TaskTabular, modules: []recipe.ModuleID{modelrecipe.ModuleTabularPredict}},
		{task: recipe.TaskSeq2Seq, modules: []recipe.ModuleID{modelrecipe.ModuleSeq2SeqGenerate}},
		{task: recipe.TaskSpeech, modules: []recipe.ModuleID{
			modelrecipe.ModuleSpeechTokenize, modelrecipe.ModuleSpeechGenerate, modelrecipe.ModuleSpeechDecode,
		}},
	}
	for _, test := range tests {
		t.Run(string(test.task), func(t *testing.T) {
			capability, ok := capabilityCommands[test.task]
			if !ok || capability.inventory == nil || capability.definition == nil || capability.execute == nil {
				t.Fatalf("capability = %+v", capability)
			}
			modelID := testutil.ArtifactID(t, artifact.KindModel, string(test.task)+"-command-model")
			definition, err := capability.definition(modelID)
			if err != nil {
				t.Fatal(err)
			}
			program, err := modelrecipe.CompileCapability(definition)
			if err != nil {
				t.Fatal(err)
			}
			stages := program.Stages()
			if len(stages) != len(test.modules) {
				t.Fatalf("program = %+v", program)
			}
			for index, module := range test.modules {
				if stages[index].Module.ID != module {
					t.Fatalf("stage[%d] = %+v", index, stages[index])
				}
			}
		})
	}
}

func TestTabularInputIsValidatedCanonicalArtifact(t *testing.T) {
	request, content, err := tabularInput(tabularInputFixture)
	if err != nil {
		t.Fatal(err)
	}
	if request.Task != tabularicl.TaskClassification || request.Rows != tabularInputRows ||
		request.Cols != tabularInputCols || request.TrainRows != tabularInputTrainRows {
		t.Fatalf("request = %+v", request)
	}
	if err := content.Validate(); err != nil || content.Descriptor.ID.Kind() != artifact.KindFile {
		t.Fatalf("input content = (%+v, %v)", content.Descriptor, err)
	}
	if _, _, err := tabularInput(invalidTabularInputFixture); err == nil {
		t.Fatal("unknown tabular input field accepted")
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

func TestSeq2SeqInputIsValidatedCanonicalArtifact(t *testing.T) {
	request, content, err := seq2seqInput(seq2seqInputFixture)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Source) != 2 || request.Source[0] != 3 || request.Source[1] != 5 || request.MaxTokens != 2 {
		t.Fatalf("request = %+v", request)
	}
	if err := content.Validate(); err != nil || content.Descriptor.ID.Kind() != artifact.KindFile {
		t.Fatalf("input content = (%+v, %v)", content.Descriptor, err)
	}
	if _, _, err := seq2seqInput(`{"source":[],"max_tokens":2}`); err == nil {
		t.Fatal("empty seq2seq source accepted")
	}
}

func TestSpeechInputIsValidatedCanonicalArtifact(t *testing.T) {
	request, content, err := speechInput(speechInputFixture)
	if err != nil {
		t.Fatal(err)
	}
	if request.Text != "hello" || request.MaxFrames != 2 || request.Seed != 7 {
		t.Fatalf("request = %+v", request)
	}
	if err := content.Validate(); err != nil || content.Descriptor.ID.Kind() != artifact.KindFile {
		t.Fatalf("input content = (%+v, %v)", content.Descriptor, err)
	}
	if _, _, err := speechInput(`{"text":"","max_frames":2}`); err == nil {
		t.Fatal("empty speech text accepted")
	}
}
