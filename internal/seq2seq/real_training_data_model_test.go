//go:build modeltest

package seq2seq

import (
	"bufio"
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

func realGSM8KTrainingPair(t *testing.T, ordinal int) (*Generator, TrainingPair, trainingdata.Example) {
	t.Helper()
	if ordinal < 0 {
		t.Fatal("negative GSM8K record ordinal")
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	recordPath := filepath.Join(roots.Datasets, "openai_gsm8k_train.jsonl")
	file, err := os.Open(recordPath)
	if err != nil {
		t.Fatalf("GSM8K dataset unavailable: %v", err)
	}
	scanner := bufio.NewScanner(file)
	for index := 0; index <= ordinal; index++ {
		if !scanner.Scan() {
			file.Close()
			t.Fatalf("GSM8K record %d unavailable: %v", ordinal, scanner.Err())
		}
	}
	record := scanner.Text()
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	datasetID, _ := artifact.IdentifyBytes(artifact.KindDataset, []byte(recordPath))
	splitID, _ := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte(recordPath+"#train"))
	processorID, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("json-text-pair/question-answer/v1"))
	processor, err := trainingdata.JSONTextPairProcessor("question", "answer")
	if err != nil {
		t.Fatal(err)
	}
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID}, Seed: 17,
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
	}
	materialized, err := trainingdata.MaterializeDocuments(authority, processorID, []string{record}, trainingdata.ProcessorBinding{
		Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityText}, Process: processor,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := batcher.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	generator, err := LoadGenerator(filepath.Join(roots.Models, "needle"))
	if err != nil {
		t.Fatal(err)
	}
	pair, err := generator.TrainingPair(batch.Examples[0])
	if err != nil {
		t.Fatal(err)
	}
	return generator, pair, batch.Examples[0]
}

func TestNeedleRealGSM8KTrainingPair(t *testing.T) {
	generator, pair, example := realGSM8KTrainingPair(t, 0)
	if len(pair.Source) < 12 || len(pair.Targets) < 20 || pair.DecoderInput[0] != generator.model.Dims.StartToken || pair.Targets[len(pair.Targets)-1] != generator.model.Dims.EOSToken {
		t.Fatalf("unexpected real pair geometry source=%d decoder=%d target=%d", len(pair.Source), len(pair.DecoderInput), len(pair.Targets))
	}
	loss, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(loss) || math.IsInf(loss, 0) || loss <= 0 || loss >= 20 {
		t.Fatalf("real GSM8K teacher-forced loss=%g", loss)
	}
	input, target, _ := trainingdata.TextPair(example)
	if !strings.Contains(input, "Natalia") || !strings.Contains(target, "#### 72") {
		t.Fatalf("unexpected GSM8K record %q -> %q", input, target)
	}
	t.Logf("real GSM8K record: source_tokens=%d target_tokens=%d baseline_loss=%.6f", len(pair.Source), len(pair.Targets), loss)
}

func TestNeedleRealGSM8KCompiledTraining(t *testing.T) {
	generator, pair, _ := realGSM8KTrainingPair(t, 0)
	before, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(generator.model, 3, 0.001, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	operators := trainer.Program().Operators()
	if len(operators) != 3 || operators[0].ID != trainingForward || operators[1].ID != trainingBackward || operators[2].ID != trainingMuon {
		t.Fatalf("compiled operators = %+v", operators)
	}
	trajectory := make([]float64, 3)
	for step := range trajectory {
		trajectory[step], err = trainer.Step(pair)
		if err != nil {
			t.Fatal(err)
		}
	}
	after, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	if trajectory[0] != before || !(trajectory[1] < trajectory[0] && trajectory[2] < trajectory[1] && after < trajectory[2]) {
		t.Fatalf("real GSM8K loss did not descend: before=%g trajectory=%v after=%g", before, trajectory, after)
	}
	t.Logf("real GSM8K compiled Muon: %.6f -> %.6f via %v", before, after, trajectory)
}

func TestNeedleRealGSM8KHeldOutEvaluation(t *testing.T) {
	generator, trainPair, _ := realGSM8KTrainingPair(t, 0)
	_, heldOutPair, heldOutExample := realGSM8KTrainingPair(t, 1)
	trainBefore, err := generator.model.Loss(trainPair)
	if err != nil {
		t.Fatal(err)
	}
	heldOutBefore, err := generator.model.Loss(heldOutPair)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(generator.model, 3, 0.001, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	for range 3 {
		if _, err := trainer.Step(trainPair); err != nil {
			t.Fatal(err)
		}
	}
	trainAfter, err := generator.model.Loss(trainPair)
	if err != nil {
		t.Fatal(err)
	}
	heldOutAfter, err := generator.model.Loss(heldOutPair)
	if err != nil {
		t.Fatal(err)
	}
	if !(trainAfter < trainBefore) || math.IsNaN(heldOutAfter) || math.IsInf(heldOutAfter, 0) || heldOutAfter <= 0 {
		t.Fatalf("invalid train/held-out result train %.6f -> %.6f held-out %.6f -> %.6f", trainBefore, trainAfter, heldOutBefore, heldOutAfter)
	}
	heldInput, heldTarget, _ := trainingdata.TextPair(heldOutExample)
	if !strings.Contains(heldInput, "babysitting") || !strings.Contains(heldTarget, "#### 10") {
		t.Fatalf("unexpected held-out GSM8K record %q -> %q", heldInput, heldTarget)
	}
	t.Logf("real GSM8K train %.6f -> %.6f; held-out %.6f -> %.6f", trainBefore, trainAfter, heldOutBefore, heldOutAfter)
}

func TestNeedleRealGSM8KFinalCrossAttentionTraining(t *testing.T) {
	generator, pair, _ := realGSM8KTrainingPair(t, 0)
	block := &generator.model.decoderCross[len(generator.model.decoderCross)-1]
	rawBefore, foldedBefore := block.rawGate, block.gate
	trainer, err := NewTrainer(generator.model, 3, 0.001, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	parameters := trainer.Program().Parameters()
	if len(parameters) != 3 || parameters[1].Name != "decoder.final_cross.raw_gate" || parameters[2].Name != "decoder.final_cross.output" {
		t.Fatalf("compiled parameters = %+v", parameters)
	}
	probe := trainingStep{pair: pair}
	if err := trainer.forward(&probe); err != nil {
		t.Fatal(err)
	}
	if err := trainer.backward(&probe); err != nil {
		t.Fatal(err)
	}
	const epsilon = float32(1e-3)
	block.rawGate, block.gate = rawBefore+epsilon, sigmoid(rawBefore+epsilon)
	plus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.rawGate, block.gate = rawBefore-epsilon, sigmoid(rawBefore-epsilon)
	minus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.rawGate, block.gate = rawBefore, foldedBefore
	finiteDifference := (plus - minus) / (2 * float64(epsilon))
	analytic := float64(probe.gradient[generator.model.Dims.DModel])
	if delta := math.Abs(analytic - finiteDifference); delta > 2e-3 {
		t.Fatalf("final cross gate gradient analytic=%g finite_difference=%g delta=%g", analytic, finiteDifference, delta)
	}
	trajectory := make([]float64, 3)
	for step := range trajectory {
		trajectory[step], err = trainer.Step(pair)
		if err != nil {
			t.Fatal(err)
		}
	}
	if block.rawGate == rawBefore || block.gate == foldedBefore {
		t.Fatalf("final cross gate did not update: raw=%g folded=%g", block.rawGate, block.gate)
	}
	after, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	if !(trajectory[1] < trajectory[0] && trajectory[2] < trajectory[1] && after < trajectory[2]) {
		t.Fatalf("final cross training did not descend: %v -> %g", trajectory, after)
	}
	t.Logf("real GSM8K final-cross Muon: %v -> %.6f; raw gate %.6f -> %.6f", trajectory, after, rawBefore, block.rawGate)
}

func TestNeedleRealGSM8KFinalCrossOutputGradient(t *testing.T) {
	generator, pair, _ := realGSM8KTrainingPair(t, 0)
	trainer, err := NewTrainer(generator.model, 1, 0.001, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	probe := trainingStep{pair: pair}
	if err := trainer.forward(&probe); err != nil {
		t.Fatal(err)
	}
	if err := trainer.backward(&probe); err != nil {
		t.Fatal(err)
	}
	block := &generator.model.decoderCross[len(generator.model.decoderCross)-1]
	index := -1
	for candidate, gradient := range probe.projectionGradient {
		word := block.o[candidate]
		if word > 1 && word < 0x7f00 && (word&0x8000) == 0 &&
			(index < 0 || math.Abs(float64(gradient)) > math.Abs(float64(probe.projectionGradient[index]))) {
			index = candidate
		}
	}
	if index < 0 {
		t.Fatal("positive final cross output weight absent")
	}
	original := block.o[index]
	block.o[index] = original + 1
	plus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.o[index] = original - 1
	minus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.o[index] = original
	plusWeight := float64(math.Float32frombits(uint32(original+1) << 16))
	minusWeight := float64(math.Float32frombits(uint32(original-1) << 16))
	finiteDifference := (plus - minus) / (plusWeight - minusWeight)
	analytic := float64(probe.projectionGradient[index])
	delta := math.Abs(analytic - finiteDifference)
	limit := 0.03 * max(math.Abs(analytic), math.Abs(finiteDifference))
	if delta > limit {
		t.Fatalf("final cross output gradient[%d] analytic=%g finite_difference=%g delta=%g limit=%g", index, analytic, finiteDifference, delta, limit)
	}
	t.Logf("real GSM8K final cross output gradient[%d]: analytic=%g finite_difference=%g", index, analytic, finiteDifference)
}

func TestNeedleRealGSM8KFinalCrossOutputMuon(t *testing.T) {
	generator, pair, _ := realGSM8KTrainingPair(t, 0)
	block := &generator.model.decoderCross[len(generator.model.decoderCross)-1]
	original := append([]uint16(nil), block.o...)
	trainer, err := NewTrainer(generator.model, 3, 0.001, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	trajectory := make([]float64, 3)
	for step := range trajectory {
		trajectory[step], err = trainer.Step(pair)
		if err != nil {
			t.Fatal(err)
		}
	}
	changed := 0
	for index, word := range block.o {
		if word != original[index] {
			changed++
		}
	}
	after, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	if changed == 0 || !(trajectory[1] < trajectory[0] && trajectory[2] < trajectory[1] && after < trajectory[2]) {
		t.Fatalf("output projection did not train: changed=%d trajectory=%v after=%g", changed, trajectory, after)
	}
	t.Logf("real GSM8K device Muon: %v -> %.6f; output BF16 words changed=%d/%d", trajectory, after, changed, len(block.o))
}
