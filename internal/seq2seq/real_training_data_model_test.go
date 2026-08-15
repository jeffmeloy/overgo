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

func TestNeedleRealGSM8KTrainingPair(t *testing.T) {
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
	if !scanner.Scan() {
		file.Close()
		t.Fatalf("GSM8K first record unavailable: %v", scanner.Err())
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
	input, target, _ := trainingdata.TextPair(batch.Examples[0])
	if !strings.Contains(input, "Natalia") || !strings.Contains(target, "#### 72") {
		t.Fatalf("unexpected GSM8K record %q -> %q", input, target)
	}
	t.Logf("real GSM8K record: source_tokens=%d target_tokens=%d baseline_loss=%.6f", len(pair.Source), len(pair.Targets), loss)
}
