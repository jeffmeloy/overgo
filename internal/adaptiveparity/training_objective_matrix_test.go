package adaptiveparity

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

const adaptiveObjectiveSource = "214950b3b0316bcdcab38a3b95127a927c0ab5be"

func TestTrainingObjectiveMatrix(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "objective-breadth"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	plan, err := optimizer.CompilePlan(4, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 4, Rows: 2, Cols: 2}})
	if err != nil {
		t.Fatal(err)
	}
	parameters := []trainingprogram.ParameterSpec{{Name: "weight", Rows: 2, Cols: 2, Trainable: true}}
	tests := []struct {
		kind          trainingprogram.ObjectiveKind
		input, output recipecontract.Modality
		metric        trainingprogram.EvaluationMetric
	}{
		{trainingprogram.ObjectiveFNS, recipecontract.ModalityText, recipecontract.ModalityText, trainingprogram.MetricTokenAccuracy},
		{trainingprogram.ObjectiveLatentL2, recipecontract.ModalityText, recipecontract.ModalityAudio, trainingprogram.MetricAudioSNR},
		{trainingprogram.ObjectiveLatentSequence, recipecontract.ModalityText, recipecontract.ModalityAudio, trainingprogram.MetricAudioSNR},
		{trainingprogram.ObjectiveForecast, recipecontract.ModalityTimeSeries, recipecontract.ModalityTimeSeries, trainingprogram.MetricForecastMAE},
		{trainingprogram.ObjectiveOCR, recipecontract.ModalityImage, recipecontract.ModalityText, trainingprogram.MetricTokenAccuracy},
		{trainingprogram.ObjectiveFlowMatching, recipecontract.ModalityText, recipecontract.ModalityAudio, trainingprogram.MetricAudioSNR},
		{trainingprogram.ObjectiveImageLatent, recipecontract.ModalityImage, recipecontract.ModalityImage, trainingprogram.MetricImagePSNRUnit},
		{trainingprogram.ObjectiveDistillation, recipecontract.ModalityText, recipecontract.ModalityText, trainingprogram.MetricTokenAccuracy},
	}
	programs := make([]trainingprogram.TrainingProgram, len(tests))
	objectives := make([]artifact.ID, len(tests))
	var descriptors []artifact.Descriptor
	var contents []artifact.Content
	for index, test := range tests {
		programs[index], err = trainingprogram.CompileObjectiveProgram(test.kind, parameters, plan)
		if err != nil {
			t.Fatalf("compile %s: %v", test.kind, err)
		}
		identity := func(kind artifact.Kind, role string) artifact.ID {
			return testutil.ArtifactID(t, kind, string(test.kind)+"/"+role)
		}
		document, objectiveErr := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
			Name: string(test.kind), Kind: test.kind,
			Signature: recipecontract.ModalitySignature{
				Inputs: []recipecontract.Modality{test.input}, Outputs: []recipecontract.Modality{test.output},
			},
			Dataset: identity(artifact.KindDataset, "dataset"), Split: identity(artifact.KindDatasetShard, "split"),
			Processors: []artifact.ID{identity(artifact.KindProfile, "processor")},
			Loss:       identity(artifact.KindProfile, "loss"), Evaluation: identity(artifact.KindProfile, "evaluation"),
			Metric: test.metric, Evidence: []artifact.ID{
				testutil.ArtifactID(t, artifact.KindEvidence, adaptiveObjectiveSource+"/"+string(test.kind)),
			},
			Authority: trainingprogram.ObjectiveAdaptive,
		})
		if objectiveErr != nil {
			t.Fatalf("objective %s: %v", test.kind, objectiveErr)
		}
		content, contentErr := document.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		contents = append(contents, content)
		objectives[index] = document.ID
		for _, id := range append([]artifact.ID{
			document.Dataset, document.Split, document.Loss, document.Evaluation,
		}, append(document.Processors, document.Evidence...)...) {
			descriptors = append(descriptors, artifact.Descriptor{ID: id})
		}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "adaptive-objective-breadth", Artifacts: descriptors, Contents: contents}); err != nil {
		t.Fatal(err)
	}
	rows, err := trainingprogram.CompileTrainingObjectiveMatrix(ctx, store, objectives, programs)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(tests) {
		t.Fatalf("objective rows = %d, want %d", len(rows), len(tests))
	}
	for _, row := range rows {
		if row.Disposition != trainingprogram.ObjectiveProgramCompiled || !row.Program.Valid() || row.Reason != "" {
			t.Fatalf("objective row = %+v", row)
		}
	}
	for _, program := range programs {
		assertObjectiveProgramOrder(t, program)
	}
	withoutDistillation := slices.DeleteFunc(slices.Clone(programs), func(program trainingprogram.TrainingProgram) bool {
		return program.Objective() == trainingprogram.ObjectiveDistillation
	})
	rows, err = trainingprogram.CompileTrainingObjectiveMatrix(ctx, store, objectives, withoutDistillation)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if row.Kind == trainingprogram.ObjectiveDistillation &&
			(row.Disposition != trainingprogram.ObjectiveProgramRefused || row.Reason == "" || row.Program.Valid()) {
			t.Fatalf("unbound distillation row = %+v", row)
		}
	}
	t.Logf("adaptive objective contracts: source=%s compiled=%d; absent binding refused", adaptiveObjectiveSource, len(rows))
}

type objectiveOrderState struct{ phases []string }

func assertObjectiveProgramOrder(t *testing.T, program trainingprogram.TrainingProgram) {
	t.Helper()
	binding := func(operator string) trainingprogram.Binding[objectiveOrderState] {
		return trainingprogram.Binding[objectiveOrderState]{Operator: operator, Execute: func(state *objectiveOrderState) error {
			state.phases = append(state.phases, operator)
			return nil
		}}
	}
	execution, err := trainingprogram.Bind(program, []trainingprogram.Binding[objectiveOrderState]{
		binding(trainingprogram.ObjectiveOperatorMuon),
		binding(trainingprogram.ObjectiveOperatorBackward),
		binding(trainingprogram.ObjectiveOperatorForward),
	})
	if err != nil {
		t.Fatal(err)
	}
	execution, err = execution.Select(trainingprogram.PhaseForward, trainingprogram.PhaseBackward, trainingprogram.PhaseOptimize)
	if err != nil {
		t.Fatal(err)
	}
	state := objectiveOrderState{}
	if err := execution.Run(&state); err != nil {
		t.Fatal(err)
	}
	want := []string{
		trainingprogram.ObjectiveOperatorForward,
		trainingprogram.ObjectiveOperatorBackward,
		trainingprogram.ObjectiveOperatorMuon,
	}
	if !slices.Equal(state.phases, want) {
		t.Fatalf("objective %s order = %v, want %v", program.Objective(), state.phases, want)
	}
}
