package trainingprogram

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestObjectiveMatrixUsesOnlyStoredCorpusBoundObjectives(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, err := repodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	text := objectiveFixture(t, recipecontract.ModalityText, recipecontract.ModalityText, false)
	image := objectiveFixture(t, recipecontract.ModalityImage, recipecontract.ModalityText, true)
	descriptors := objectiveReferences(text)
	descriptors = append(descriptors, objectiveReferences(image)...)
	contents := make([]artifact.Content, 0, 2)
	for _, objective := range []ObjectiveDocument{text, image} {
		content, err := objective.Content()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, content)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "objectives", Artifacts: descriptors, Contents: contents}); err != nil {
		t.Fatal(err)
	}
	rows, err := CompileObjectiveMatrix(ctx, store, []artifact.ID{image.ID, text.ID})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 36 {
		t.Fatalf("objective rows=%d, want 36", len(rows))
	}
	trainable := map[string]artifact.ID{"text->text": text.ID, "image->text": image.ID}
	for _, row := range rows {
		key := objectivePairKey(row.Signature)
		want, ok := trainable[key]
		if ok {
			if row.Disposition != ObjectiveTrainable || row.Objective != want || row.Reason != "" {
				t.Fatalf("trainable row %s differs: %+v", key, row)
			}
			delete(trainable, key)
			continue
		}
		if row.Disposition != ObjectiveRefused || row.Objective.Valid() || row.Reason == "" {
			t.Fatalf("refusal row %s differs: %+v", key, row)
		}
	}
	if len(trainable) != 0 {
		t.Fatalf("trainable rows absent: %v", trainable)
	}
}

func TestRepositoryObjectiveBindsTrainingRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, err := repodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objective := objectiveFixture(t, recipecontract.ModalityImage, recipecontract.ModalityText, true)
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "run-objective", Artifacts: objectiveReferences(objective), Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	muon, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileTrainingProgram(ProgramSpec{
		Operators:  []OperatorSpec{{ID: "forward", Phase: PhaseForward}, {ID: "backward", Phase: PhaseBackward}, {ID: "muon", Phase: PhaseOptimize}},
		Parameters: []ParameterSpec{{Name: "weight", Rows: 1, Cols: 1, Trainable: true}}, Optimizer: muon,
	})
	if err != nil {
		t.Fatal(err)
	}
	identity := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	profile := func(name string) artifact.ID { return identity(artifact.KindProfile, name) }
	spec := RunSpec{
		Recipe:  identity(artifact.KindRecipe, "run"),
		Initial: InitialStateSpec{Model: identity(artifact.KindModel, "model")},
		Dataset: objective.Dataset, Split: objective.Split, Signature: objective.Signature,
		Processors: objective.Processors, Projectors: objective.Projectors, Codecs: objective.Codecs,
		Policies: PolicySpec{
			Objective: objective.ID, Precision: profile("precision"), Placement: profile("placement"),
			Memory: profile("memory"), Checkpoint: profile("checkpoint"), Evaluation: objective.Evaluation,
			Promotion: profile("promotion"),
		},
		Program: program,
	}
	if _, err := CompileTrainingRunPlanFromRepository(ctx, store, spec); err != nil {
		t.Fatal(err)
	}
	spec.Signature.Inputs[0] = recipecontract.ModalityAudio
	if _, err := CompileTrainingRunPlanFromRepository(ctx, store, spec); err == nil {
		t.Fatal("run with a different modality signature accepted")
	}
}

func TestObjectiveMatrixRefusesUnstoredEvidence(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, err := repodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objective := objectiveFixture(t, recipecontract.ModalityText, recipecontract.ModalityAudio, false)
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "objective-only", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	if _, err := CompileObjectiveMatrix(ctx, store, []artifact.ID{objective.ID}); err == nil {
		t.Fatal("objective with unstored corpus/evidence references accepted")
	}
}

func objectiveFixture(t *testing.T, input, output recipecontract.Modality, projector bool) ObjectiveDocument {
	t.Helper()
	ids := func(kind artifact.Kind, label string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(label+"/"+string(input)+"/"+string(output)))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	var projectors []artifact.ID
	if projector {
		projectors = []artifact.ID{ids(artifact.KindProjector, "projector")}
	}
	objective, err := NewObjective(ObjectiveSpec{
		Name: string(input) + "-to-" + string(output),
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{input}, Outputs: []recipecontract.Modality{output},
		},
		Dataset: ids(artifact.KindDataset, "dataset"), Split: ids(artifact.KindDatasetShard, "split"),
		Processors: []artifact.ID{ids(artifact.KindProfile, "processor")}, Projectors: projectors,
		Loss: ids(artifact.KindProfile, "loss"), Evaluation: ids(artifact.KindProfile, "evaluation"),
		Metric:   fixtureMetric(output),
		Evidence: []artifact.ID{ids(artifact.KindEvidence, "evidence")}, Authority: ObjectiveAdaptive,
	})
	if err != nil {
		t.Fatal(err)
	}
	return objective
}

func objectiveReferences(objective ObjectiveDocument) []artifact.Descriptor {
	ids := []artifact.ID{objective.Dataset, objective.Split, objective.Loss, objective.Evaluation}
	ids = append(ids, objective.Processors...)
	ids = append(ids, objective.Projectors...)
	ids = append(ids, objective.Codecs...)
	ids = append(ids, objective.Evidence...)
	result := make([]artifact.Descriptor, len(ids))
	for index, id := range ids {
		result[index] = artifact.Descriptor{ID: id}
	}
	return result
}

func fixtureMetric(modality recipecontract.Modality) EvaluationMetric {
	switch modality {
	case recipecontract.ModalityText:
		return MetricTokenAccuracy
	case recipecontract.ModalityImage:
		return MetricImagePSNRSigned
	case recipecontract.ModalityAudio:
		return MetricAudioSNR
	case recipecontract.ModalityVideo:
		return MetricVideoPSNRSigned
	case recipecontract.ModalityTimeSeries:
		return MetricForecastMAE
	case recipecontract.ModalityTable:
		return MetricTableAccuracy
	default:
		return ""
	}
}
