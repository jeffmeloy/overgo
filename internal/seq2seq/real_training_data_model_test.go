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
	probe := trainingStep{pair: pair}
	if err := trainer.forward(&probe); err != nil {
		t.Fatal(err)
	}
	if len(probe.trace.layers) != generator.model.Dims.DecoderLayers {
		t.Fatalf("decoder traces=%d want=%d", len(probe.trace.layers), generator.model.Dims.DecoderLayers)
	}
	for layer, trace := range probe.trace.layers {
		if len(trace.self.projected) == 0 || len(trace.cross.projected) == 0 || len(trace.cross.source) == 0 {
			t.Fatalf("decoder layer %d training trace is incomplete", layer)
		}
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
	if len(parameters) != 25 || parameters[1].Name != "decoder.final_cross.raw_gate" ||
		parameters[2].Name != "decoder.final_cross.output" || parameters[24].Name != "decoder.layer6.cross.v" {
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

func TestNeedleRealGSM8KFinalCrossAttentionCoreGradient(t *testing.T) {
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
	norm := func(values []float32) float64 {
		var sum float64
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("non-finite cross-attention gradient")
			}
			sum += float64(value) * float64(value)
		}
		return math.Sqrt(sum)
	}
	qNorm, kNorm, vNorm := norm(probe.attentionQGradient), norm(probe.attentionKGradient), norm(probe.attentionVGradient)
	if qNorm == 0 || kNorm == 0 || vNorm == 0 {
		t.Fatalf("real cross-attention gradient norms q=%g k=%g v=%g", qNorm, kNorm, vNorm)
	}
	t.Logf("real GSM8K final cross-attention gradient norms: q=%g k=%g v=%g", qNorm, kNorm, vNorm)
}

func TestNeedleRealGSM8KFinalCrossQKVMuon(t *testing.T) {
	generator, trainPair, _ := realGSM8KTrainingPair(t, 0)
	_, heldOutPair, _ := realGSM8KTrainingPair(t, 1)
	block := &generator.model.decoderCross[len(generator.model.decoderCross)-1]
	qBefore := append([]uint16(nil), block.q...)
	kBefore := append([]uint16(nil), block.k...)
	vBefore := append([]uint16(nil), block.v...)
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
	wantParameters := []string{
		"decoder.final_norm", "decoder.final_cross.raw_gate", "decoder.final_cross.output",
		"decoder.final_cross.input_norm", "decoder.final_cross.q_norm", "decoder.final_cross.k_norm",
		"decoder.final_cross.q", "decoder.final_cross.k", "decoder.final_cross.v",
		"decoder.final_self.raw_gate", "decoder.final_self.output", "decoder.final_self.input_norm",
		"decoder.final_self.q_norm", "decoder.final_self.k_norm",
		"decoder.final_self.q", "decoder.final_self.k", "decoder.final_self.v",
		"decoder.layer6.cross.raw_gate", "decoder.layer6.cross.output",
		"decoder.layer6.cross.input_norm", "decoder.layer6.cross.q_norm", "decoder.layer6.cross.k_norm",
		"decoder.layer6.cross.q", "decoder.layer6.cross.k", "decoder.layer6.cross.v",
	}
	parameters := trainer.Program().Parameters()
	if len(parameters) != len(wantParameters) {
		t.Fatalf("compiled parameter count=%d want=%d", len(parameters), len(wantParameters))
	}
	for index, name := range wantParameters {
		if parameters[index].Name != name || !parameters[index].Trainable {
			t.Fatalf("compiled parameter[%d]=%+v want trainable %q", index, parameters[index], name)
		}
	}
	probe := trainingStep{pair: trainPair}
	if err := trainer.forward(&probe); err != nil {
		t.Fatal(err)
	}
	if err := trainer.backward(&probe); err != nil {
		t.Fatal(err)
	}
	for name, gradient := range map[string][]float32{
		"q": probe.qProjectionGradient, "k": probe.kProjectionGradient, "v": probe.vProjectionGradient,
	} {
		if gradientL2(gradient) == 0 {
			t.Fatalf("final cross %s projection gradient is zero", name)
		}
	}
	qAnalytic, qFinite := verifyBF16Gradient(t, generator.model, trainPair, block.q, probe.qProjectionGradient)
	kAnalytic, kFinite := verifyBF16Gradient(t, generator.model, trainPair, block.k, probe.kProjectionGradient)
	vAnalytic, vFinite := verifyBF16Gradient(t, generator.model, trainPair, block.v, probe.vProjectionGradient)
	inAnalytic, inFinite := verifyFloat32Gradient(t, generator.model, trainPair, block.inNorm, probe.gradient[trainer.layout.inputNorm.start:trainer.layout.inputNorm.end])
	qNormAnalytic, qNormFinite := verifyFloat32Gradient(t, generator.model, trainPair, block.qNorm, probe.gradient[trainer.layout.qNorm.start:trainer.layout.qNorm.end])
	kNormAnalytic, kNormFinite := verifyFloat32Gradient(t, generator.model, trainPair, block.kNorm, probe.gradient[trainer.layout.kNorm.start:trainer.layout.kNorm.end])
	trajectory := make([]float64, 3)
	for step := range trajectory {
		trajectory[step], err = trainer.Step(trainPair)
		if err != nil {
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
	qChanged := changedBF16(qBefore, block.q)
	kChanged := changedBF16(kBefore, block.k)
	vChanged := changedBF16(vBefore, block.v)
	if qChanged == 0 || kChanged == 0 || vChanged == 0 ||
		!(trajectory[1] < trajectory[0] && trajectory[2] < trajectory[1] && trainAfter < trajectory[2]) ||
		math.IsNaN(heldOutAfter) || math.IsInf(heldOutAfter, 0) {
		t.Fatalf("final cross Q/K/V did not train: changed=%d/%d/%d train %.6f -> %v -> %.6f held-out %.6f -> %.6f",
			qChanged, kChanged, vChanged, trainBefore, trajectory, trainAfter, heldOutBefore, heldOutAfter)
	}
	t.Logf("real GSM8K final cross Q/K/V Muon: train %.6f -> %.6f via %v; held-out %.6f -> %.6f; BF16 changed q=%d/%d k=%d/%d v=%d/%d",
		trainBefore, trainAfter, trajectory, heldOutBefore, heldOutAfter,
		qChanged, len(block.q), kChanged, len(block.k), vChanged, len(block.v))
	t.Logf("real GSM8K final cross finite differences: q=%g/%g k=%g/%g v=%g/%g input_norm=%g/%g q_norm=%g/%g k_norm=%g/%g",
		qAnalytic, qFinite, kAnalytic, kFinite, vAnalytic, vFinite,
		inAnalytic, inFinite, qNormAnalytic, qNormFinite, kNormAnalytic, kNormFinite)
}

func TestNeedleRealGSM8KFinalSelfAttentionGradient(t *testing.T) {
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
	qNorm := gradientL2(probe.selfAttentionQGradient)
	kNorm := gradientL2(probe.selfAttentionKGradient)
	vNorm := gradientL2(probe.selfAttentionVGradient)
	if qNorm == 0 || kNorm == 0 || vNorm == 0 {
		t.Fatalf("real final self-attention gradient norms q=%g k=%g v=%g", qNorm, kNorm, vNorm)
	}
	block := &generator.model.decoderSelf[len(generator.model.decoderSelf)-1]
	originalRaw, originalGate := block.rawGate, block.gate
	const epsilon = float32(1e-3)
	block.rawGate, block.gate = originalRaw+epsilon, sigmoid(originalRaw+epsilon)
	plus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.rawGate, block.gate = originalRaw-epsilon, sigmoid(originalRaw-epsilon)
	minus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.rawGate, block.gate = originalRaw, originalGate
	finite := (plus - minus) / (2 * float64(epsilon))
	analytic := float64(probe.selfRawGateGradient)
	if delta := math.Abs(analytic - finite); delta > 2e-3 {
		t.Fatalf("final self gate gradient analytic=%g finite_difference=%g delta=%g", analytic, finite, delta)
	}
	t.Logf("real GSM8K final self-attention gradients: q=%g k=%g v=%g; raw gate analytic=%g finite_difference=%g",
		qNorm, kNorm, vNorm, analytic, finite)
}

func TestNeedleRealGSM8KPenultimateCrossAttentionGradient(t *testing.T) {
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
	qNorm, kNorm, vNorm := gradientL2(probe.penultimateCrossQ), gradientL2(probe.penultimateCrossK), gradientL2(probe.penultimateCrossV)
	if qNorm == 0 || kNorm == 0 || vNorm == 0 {
		t.Fatalf("penultimate cross-attention gradient norms q=%g k=%g v=%g", qNorm, kNorm, vNorm)
	}
	layer := generator.model.Dims.DecoderLayers - 2
	block := &generator.model.decoderCross[layer]
	originalRaw, originalGate := block.rawGate, block.gate
	const epsilon = float32(1e-3)
	block.rawGate, block.gate = originalRaw+epsilon, sigmoid(originalRaw+epsilon)
	plus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.rawGate, block.gate = originalRaw-epsilon, sigmoid(originalRaw-epsilon)
	minus, err := generator.model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	block.rawGate, block.gate = originalRaw, originalGate
	finite := (plus - minus) / (2 * float64(epsilon))
	analytic := float64(probe.penultimateCrossRawGate)
	if delta := math.Abs(analytic - finite); delta > 2e-3 {
		t.Fatalf("penultimate cross gate gradient analytic=%g finite_difference=%g delta=%g", analytic, finite, delta)
	}
	t.Logf("real GSM8K decoder layer %d cross gradients: q=%g k=%g v=%g; raw gate analytic=%g finite_difference=%g",
		layer, qNorm, kNorm, vNorm, analytic, finite)
}

func TestNeedleRealGSM8KPenultimateCrossAttentionMuon(t *testing.T) {
	generator, trainPair, _ := realGSM8KTrainingPair(t, 0)
	_, heldOutPair, _ := realGSM8KTrainingPair(t, 1)
	layer := generator.model.Dims.DecoderLayers - 2
	block := &generator.model.decoderCross[layer]
	outputBefore := append([]uint16(nil), block.o...)
	gateBefore := block.rawGate
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
	trajectory := make([]float64, 3)
	for step := range trajectory {
		trajectory[step], err = trainer.Step(trainPair)
		if err != nil {
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
	changed := changedBF16(outputBefore, block.o)
	if changed == 0 || block.rawGate == gateBefore ||
		!(trajectory[1] < trajectory[0] && trajectory[2] < trajectory[1] && trainAfter < trajectory[2]) ||
		math.IsNaN(heldOutAfter) || math.IsInf(heldOutAfter, 0) {
		t.Fatalf("layer %d cross boundary did not train: output_changed=%d gate=%g/%g train %.6f -> %v -> %.6f held-out %.6f -> %.6f",
			layer, changed, gateBefore, block.rawGate, trainBefore, trajectory, trainAfter, heldOutBefore, heldOutAfter)
	}
	t.Logf("real GSM8K decoder layer %d cross gate/output Muon: train %.6f -> %.6f via %v; held-out %.6f -> %.6f; output BF16 changed=%d/%d",
		layer, trainBefore, trainAfter, trajectory, heldOutBefore, heldOutAfter, changed, len(block.o))
}

func TestNeedleRealGSM8KPenultimateCrossQKVMuon(t *testing.T) {
	generator, trainPair, _ := realGSM8KTrainingPair(t, 0)
	_, heldOutPair, _ := realGSM8KTrainingPair(t, 1)
	layer := generator.model.Dims.DecoderLayers - 2
	block := &generator.model.decoderCross[layer]
	qBefore := append([]uint16(nil), block.q...)
	kBefore := append([]uint16(nil), block.k...)
	vBefore := append([]uint16(nil), block.v...)
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
	probe := trainingStep{pair: trainPair}
	if err := trainer.forward(&probe); err != nil {
		t.Fatal(err)
	}
	if err := trainer.backward(&probe); err != nil {
		t.Fatal(err)
	}
	qAnalytic, qFinite := verifyBF16GradientRadius(t, generator.model, trainPair, block.q, probe.penultimateCrossQProjection, 4)
	kAnalytic, kFinite := verifyBF16GradientRadius(t, generator.model, trainPair, block.k, probe.penultimateCrossKProjection, 4)
	vAnalytic, vFinite := verifyBF16GradientRadius(t, generator.model, trainPair, block.v, probe.penultimateCrossVProjection, 4)
	inAnalytic, inFinite := verifyFloat32Gradient(t, generator.model, trainPair, block.inNorm, probe.gradient[trainer.layout.penultimateCrossInputNorm.start:trainer.layout.penultimateCrossInputNorm.end])
	qNormAnalytic, qNormFinite := verifyFloat32Gradient(t, generator.model, trainPair, block.qNorm, probe.gradient[trainer.layout.penultimateCrossQNorm.start:trainer.layout.penultimateCrossQNorm.end])
	kNormAnalytic, kNormFinite := verifyFloat32Gradient(t, generator.model, trainPair, block.kNorm, probe.gradient[trainer.layout.penultimateCrossKNorm.start:trainer.layout.penultimateCrossKNorm.end])
	trajectory := make([]float64, 3)
	for step := range trajectory {
		trajectory[step], err = trainer.Step(trainPair)
		if err != nil {
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
	qChanged, kChanged, vChanged := changedBF16(qBefore, block.q), changedBF16(kBefore, block.k), changedBF16(vBefore, block.v)
	if qChanged == 0 || kChanged == 0 || vChanged == 0 ||
		!(trajectory[1] < trajectory[0] && trajectory[2] < trajectory[1] && trainAfter < trajectory[2]) ||
		math.IsNaN(heldOutAfter) || math.IsInf(heldOutAfter, 0) {
		t.Fatalf("layer %d cross Q/K/V did not train: changed=%d/%d/%d train %.6f -> %v -> %.6f held-out %.6f -> %.6f",
			layer, qChanged, kChanged, vChanged, trainBefore, trajectory, trainAfter, heldOutBefore, heldOutAfter)
	}
	t.Logf("real GSM8K decoder layer %d cross Q/K/V Muon: train %.6f -> %.6f via %v; held-out %.6f -> %.6f; BF16 changed q=%d/%d k=%d/%d v=%d/%d",
		layer, trainBefore, trainAfter, trajectory, heldOutBefore, heldOutAfter,
		qChanged, len(block.q), kChanged, len(block.k), vChanged, len(block.v))
	t.Logf("real GSM8K decoder layer %d finite differences: q=%g/%g k=%g/%g v=%g/%g input_norm=%g/%g q_norm=%g/%g k_norm=%g/%g",
		layer, qAnalytic, qFinite, kAnalytic, kFinite, vAnalytic, vFinite,
		inAnalytic, inFinite, qNormAnalytic, qNormFinite, kNormAnalytic, kNormFinite)
}

func TestNeedleRealGSM8KFinalSelfAttentionMuon(t *testing.T) {
	generator, trainPair, _ := realGSM8KTrainingPair(t, 0)
	_, heldOutPair, _ := realGSM8KTrainingPair(t, 1)
	block := &generator.model.decoderSelf[len(generator.model.decoderSelf)-1]
	qBefore := append([]uint16(nil), block.q...)
	kBefore := append([]uint16(nil), block.k...)
	vBefore := append([]uint16(nil), block.v...)
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
	probe := trainingStep{pair: trainPair}
	if err := trainer.forward(&probe); err != nil {
		t.Fatal(err)
	}
	if err := trainer.backward(&probe); err != nil {
		t.Fatal(err)
	}
	qAnalytic, qFinite := verifyBF16Gradient(t, generator.model, trainPair, block.q, probe.selfQGradient)
	kAnalytic, kFinite := verifyBF16Gradient(t, generator.model, trainPair, block.k, probe.selfKGradient)
	vAnalytic, vFinite := verifyBF16Gradient(t, generator.model, trainPair, block.v, probe.selfVGradient)
	trajectory := make([]float64, 3)
	for step := range trajectory {
		trajectory[step], err = trainer.Step(trainPair)
		if err != nil {
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
	qChanged := changedBF16(qBefore, block.q)
	kChanged := changedBF16(kBefore, block.k)
	vChanged := changedBF16(vBefore, block.v)
	if qChanged == 0 || kChanged == 0 || vChanged == 0 ||
		!(trajectory[1] < trajectory[0] && trajectory[2] < trajectory[1] && trainAfter < trajectory[2]) ||
		math.IsNaN(heldOutAfter) || math.IsInf(heldOutAfter, 0) {
		t.Fatalf("final self Q/K/V did not train: changed=%d/%d/%d train %.6f -> %v -> %.6f held-out %.6f -> %.6f",
			qChanged, kChanged, vChanged, trainBefore, trajectory, trainAfter, heldOutBefore, heldOutAfter)
	}
	t.Logf("real GSM8K final self Muon: train %.6f -> %.6f via %v; held-out %.6f -> %.6f; BF16 changed q=%d/%d k=%d/%d v=%d/%d",
		trainBefore, trainAfter, trajectory, heldOutBefore, heldOutAfter,
		qChanged, len(block.q), kChanged, len(block.k), vChanged, len(block.v))
	t.Logf("real GSM8K final self finite differences: q=%g/%g k=%g/%g v=%g/%g",
		qAnalytic, qFinite, kAnalytic, kFinite, vAnalytic, vFinite)
}

func gradientL2(values []float32) float64 {
	var sum float64
	for _, value := range values {
		sum += float64(value) * float64(value)
	}
	return math.Sqrt(sum)
}

func changedBF16(before, after []uint16) int {
	changed := 0
	for index, word := range after {
		if word != before[index] {
			changed++
		}
	}
	return changed
}

func verifyBF16Gradient(t *testing.T, model *Model, pair TrainingPair, weights []uint16, gradient []float32) (float64, float64) {
	return verifyBF16GradientRadius(t, model, pair, weights, gradient, 1)
}

func verifyBF16GradientRadius(t *testing.T, model *Model, pair TrainingPair, weights []uint16, gradient []float32, radius uint16) (float64, float64) {
	t.Helper()
	index := strongestFiniteGradient(gradient)
	if index < 0 {
		t.Fatal("finite BF16 gradient absent")
	}
	original := weights[index]
	if original <= radius || original >= 0xff7f-radius {
		t.Fatalf("BF16 gradient probe[%d] has unusable word %#x", index, original)
	}
	firstWord, secondWord := original-radius, original+radius
	firstValue := float64(math.Float32frombits(uint32(firstWord) << 16))
	secondValue := float64(math.Float32frombits(uint32(secondWord) << 16))
	weights[index] = firstWord
	firstLoss, err := model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	weights[index] = secondWord
	secondLoss, err := model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	weights[index] = original
	finite := (secondLoss - firstLoss) / (secondValue - firstValue)
	analytic := float64(gradient[index])
	verifyGradientClose(t, index, analytic, finite, 0.15)
	return analytic, finite
}

func verifyFloat32Gradient(t *testing.T, model *Model, pair TrainingPair, weights, gradient []float32) (float64, float64) {
	t.Helper()
	index := strongestFiniteGradient(gradient)
	if index < 0 {
		t.Fatal("finite float32 gradient absent")
	}
	original := weights[index]
	const epsilon = float32(1e-3)
	weights[index] = original + epsilon
	plus, err := model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	weights[index] = original - epsilon
	minus, err := model.Loss(pair)
	if err != nil {
		t.Fatal(err)
	}
	weights[index] = original
	finite := (plus - minus) / (2 * float64(epsilon))
	analytic := float64(gradient[index])
	verifyGradientClose(t, index, analytic, finite, 0.04)
	return analytic, finite
}

func strongestFiniteGradient(gradient []float32) int {
	index := -1
	for candidate, value := range gradient {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			continue
		}
		if index < 0 || math.Abs(float64(value)) > math.Abs(float64(gradient[index])) {
			index = candidate
		}
	}
	return index
}

func verifyGradientClose(t *testing.T, index int, analytic, finite, relativeLimit float64) {
	t.Helper()
	delta := math.Abs(analytic - finite)
	limit := relativeLimit*max(math.Abs(analytic), math.Abs(finite)) + 1e-5
	if delta > limit {
		t.Fatalf("gradient[%d] analytic=%g finite_difference=%g delta=%g limit=%g", index, analytic, finite, delta, limit)
	}
}
