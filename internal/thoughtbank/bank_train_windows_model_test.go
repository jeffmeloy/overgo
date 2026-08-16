//go:build windows && modeltest

package thoughtbank

import (
	"bufio"
	"context"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hfbpe"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

type hostBankStepper struct{ optimizer *optimizer.Optimizer }

func (stepper hostBankStepper) Step() error {
	stepper.optimizer.Step()
	return nil
}

func (hostBankStepper) Close() error { return nil }

func TestFractaleRealTrainingParity(t *testing.T) {
	cudatest.Require(t)
	modelDirectory := os.Getenv("OVERGO_THOUGHTBANK_MODEL")
	if modelDirectory == "" {
		modelDirectory = `C:\Users\jeffm\adaptive_new\models\Fractale-350M-base`
	}
	datasetPath := os.Getenv("OVERGO_WIKITEXT_DATASET")
	if datasetPath == "" {
		datasetPath = `C:\Users\jeffm\overgo\build\datasets\wikitext-2\train.txt`
	}
	for _, path := range []string{filepath.Join(modelDirectory, "model.pt"), datasetPath} {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("real Fractale training input unavailable: %v", err)
		}
	}
	weights, _, err := LoadCheckpoint(filepath.Join(modelDirectory, "model.pt"))
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := hfbpe.Load(modelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	text := orderedWikiTextRecord(t, datasetPath)
	tokens, err := tokenizer.Encode(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) < 4 {
		t.Fatalf("ordered WikiText record produced %d tokens, want at least 4", len(tokens))
	}

	const rows, slots = 2, 1
	d, mem := weights.DModel, weights.MemDim
	example := BankTrainingExample{
		Input: make([]float32, rows*d), Target: make([]float32, rows*d),
		Memory: make([]float32, slots*mem), Rows: rows, Slots: slots,
	}
	for row := range rows {
		copy(example.Input[row*d:(row+1)*d], weights.Embed[tokens[row]*d:(tokens[row]+1)*d])
		copy(example.Target[row*d:(row+1)*d], weights.Embed[tokens[row+1]*d:(tokens[row+1]+1)*d])
	}
	copy(example.Memory, weights.Embed[tokens[3]*d:tokens[3]*d+mem])

	deviceWeights := cloneBankWeights(weights.Blocks[0].Bank)
	hostWeights := cloneBankWeights(weights.Blocks[0].Bank)
	original := cloneBankWeights(weights.Blocks[0].Bank)
	config := optimizer.Config{BaseLearningRate: 1e-5, Momentum: 0.9, Schedule: optimizer.ScheduleConstant}
	deviceTrainer, err := deviceWeights.NewTrainer(config)
	if err != nil {
		t.Fatal(err)
	}
	defer deviceTrainer.Close()
	hostTrainer, err := newFastWeightBankTrainer(hostWeights, config, func(pack *optimizer.TensorPack, config optimizer.Config) (optimizer.Stepper, error) {
		host, err := pack.NewOptimizer(config)
		if err != nil {
			return nil, err
		}
		return hostBankStepper{optimizer: host}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer hostTrainer.Close()

	deviceResult, err := deviceTrainer.Step(example)
	if err != nil {
		t.Fatal(err)
	}
	hostResult, err := hostTrainer.Step(example)
	if err != nil {
		t.Fatal(err)
	}
	if deviceResult.ProgramID == "" || deviceResult.ProgramID != hostResult.ProgramID {
		t.Fatalf("compiled program identity differs: device=%q host=%q", deviceResult.ProgramID, hostResult.ProgramID)
	}
	if delta := math.Abs(deviceResult.Loss - hostResult.Loss); delta > 1e-9 {
		t.Fatalf("loss delta %.3e", delta)
	}
	maximum, changed := compareBankWeights(deviceWeights, hostWeights, original)
	t.Logf("Fractale bank update: record=%q tokens=%d loss=%.8f max device/host delta=%.3e changed=%d",
		text, len(tokens), deviceResult.Loss, maximum, changed)
	if maximum > 2e-5 {
		t.Fatalf("device/host update delta %.3e exceeds 2e-5", maximum)
	}
	if changed == 0 {
		t.Fatal("complete bank update changed no parameters")
	}
}

func orderedWikiTextRecord(t *testing.T, path string) string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	records := make([]trainingdata.RawRecord, 0, 8)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() && len(records) < cap(records) {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			records = append(records, trainingdata.RawRecord{ID: "wikitext/" + string(rune('a'+len(records))), Data: []byte(line)})
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	datasetID := trainingArtifactID(t, artifact.KindDataset, "wikitext-2-raw")
	splitID := trainingArtifactID(t, artifact.KindDatasetShard, "ordered-head")
	processorID := trainingArtifactID(t, artifact.KindProfile, "utf8-text")
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID}, Shuffle: false,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
	}
	dataset, err := trainingdata.MaterializeRecords(authority, processorID, records, trainingdata.ProcessorBinding{
		Artifact: processorID, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, trainingdata.EncodingUTF8),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer dataset.Close()
	stream, err := trainingdata.NewStream(dataset, nil)
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
	return string(batch.Examples[0].Values[0].Data)
}

func trainingArtifactID(t *testing.T, kind artifact.Kind, name string) artifact.ID {
	t.Helper()
	id, err := artifact.JSONID(kind, map[string]string{"name": name})
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func cloneBankWeights(source *FastWeightBankWeights) *FastWeightBankWeights {
	clone := *source
	clone.FWA = slices.Clone(source.FWA)
	clone.FWB = slices.Clone(source.FWB)
	clone.FWO = slices.Clone(source.FWO)
	clone.NormWeight = slices.Clone(source.NormWeight)
	return &clone
}

func compareBankWeights(device, host, original *FastWeightBankWeights) (float64, int) {
	var maximum float64
	changed := 0
	for _, tensors := range [][3][]float32{
		{device.FWA, host.FWA, original.FWA},
		{device.FWB, host.FWB, original.FWB},
		{device.FWO, host.FWO, original.FWO},
		{device.NormWeight, host.NormWeight, original.NormWeight},
	} {
		for index := range tensors[0] {
			maximum = max(maximum, math.Abs(float64(tensors[0][index])-float64(tensors[1][index])))
			if tensors[0][index] != tensors[2][index] {
				changed++
			}
		}
	}
	return maximum, changed
}
