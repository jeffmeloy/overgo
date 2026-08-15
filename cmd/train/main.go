// Command train runs Muon training on a dense causal-LM
// safetensors artifact and writes a trained checkpoint. It is the production
// caller for the densecausal training lane: it loads a model + tokenizer,
// materializes a UTF-8 dataset stream, runs ordered training updates on
// the GPU when available (host fallback), and persists the result via
// safetensors.Save. Without a caller like this the lane is exercised only by
// tests -- present but inert.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/trainingdata"
)

func main() {
	clioptions.MainNamed("train", run)
}

func run() error {
	modelDir := flag.String("model", "", "model directory (safetensors + config.json + tokenizer.json)")
	datasetPath := flag.String("dataset", "", "UTF-8 training dataset")
	outDir := flag.String("out", "", "output directory for the trained checkpoint")
	steps := flag.Int("steps", 1, "number of Muon update steps")
	maxSeq := flag.Int("seq", 512, "cap the token sequence to this length (attention is O(seq^2)); <=0 keeps all")
	baseLR := flag.Float64("lr", 0, "base learning rate; <=0 derives n_params^-1/2")
	mu := flag.Float64("momentum", 0.9, "Muon momentum")
	host := flag.Bool("host", false, "force the host path even when CUDA is available")
	freezeLexical := flag.Bool("freeze-lexical", false, "freeze tied embedding/head; requires CUDA resident training")
	flag.Parse()

	if *modelDir == "" || *datasetPath == "" || *outDir == "" {
		flag.Usage()
		return fmt.Errorf("-model, -dataset, and -out are required")
	}
	if *steps < 1 {
		return fmt.Errorf("-steps must be >= 1")
	}

	model, err := densecausal.Load(*modelDir)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	tok, err := hfbpe.Load(*modelDir)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}
	raw, err := os.ReadFile(*datasetPath)
	if err != nil {
		return fmt.Errorf("read dataset: %w", err)
	}
	batches, streamState, err := tokenBatches(context.Background(), raw, *steps, *maxSeq, tok.Encode)
	if err != nil {
		return err
	}

	if *host && *freezeLexical {
		return fmt.Errorf("-host and -freeze-lexical are mutually exclusive")
	}
	traj, backend, err := runTraining(model, batches, *baseLR, *mu, !*host, *freezeLexical)
	if err != nil {
		return fmt.Errorf("train: %w", err)
	}
	if err := saveCheckpoint(*modelDir, *outDir, model); err != nil {
		return fmt.Errorf("save checkpoint: %w", err)
	}

	fmt.Printf("backend=%s batches=%d stream_position=%d steps=%d lr=%s momentum=%g freeze_lexical=%v\n", backend, len(batches), streamState.Position, *steps, lrLabel(*baseLR), *mu, *freezeLexical)
	for i, loss := range traj {
		fmt.Printf("step %d: loss %.6f\n", i, loss)
	}
	fmt.Printf("checkpoint written to %s\n", *outDir)
	return nil
}

func lrLabel(lr float64) string {
	if lr <= 0 {
		return "derived(n^-1/2)"
	}
	return fmt.Sprintf("%g", lr)
}

func runHostTraining(
	m *densecausal.Model,
	batches [][]int,
	baseLR, mu float64,
) ([]float64, string, error) {
	trajectory, err := m.TrainBatches(batches, baseLR, mu)
	return trajectory, "host", err
}

func tokenBatches(
	ctx context.Context,
	raw []byte,
	steps, maxSeq int,
	encode func(string) ([]int, error),
) ([][]int, trainingdata.StreamState, error) {
	if len(raw) == 0 || steps <= 0 || encode == nil {
		return nil, trainingdata.StreamState{}, errors.New("train: invalid dataset stream input")
	}
	datasetID, err := artifact.IdentifyBytes(artifact.KindDataset, raw)
	if err != nil {
		return nil, trainingdata.StreamState{}, err
	}
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), raw...))
	if err != nil {
		return nil, trainingdata.StreamState{}, err
	}
	processorID, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/text-utf8/v1"))
	if err != nil {
		return nil, trainingdata.StreamState{}, err
	}
	authority := trainingdata.Authority{
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
	}
	materialized, err := trainingdata.MaterializeDocuments(authority, processorID, []string{string(raw)}, trainingdata.ProcessorBinding{
		Artifact:   processorID,
		Modalities: []recipecontract.Modality{recipecontract.ModalityText},
		Process:    trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, "utf-8"),
	})
	if err != nil {
		return nil, trainingdata.StreamState{}, err
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		return nil, trainingdata.StreamState{}, err
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1})
	if err != nil {
		return nil, trainingdata.StreamState{}, err
	}
	batches := make([][]int, steps)
	for step := range steps {
		batch, err := batcher.Next(ctx)
		if err != nil {
			return nil, trainingdata.StreamState{}, err
		}
		tokens, err := encode(string(batch.Examples[0].Values[0].Data))
		if err != nil {
			return nil, trainingdata.StreamState{}, fmt.Errorf("train: encode dataset record: %w", err)
		}
		if maxSeq > 0 && len(tokens) > maxSeq {
			tokens = tokens[:maxSeq]
		}
		if len(tokens) < 2 {
			return nil, trainingdata.StreamState{}, fmt.Errorf("train: record needs at least two tokens, got %d", len(tokens))
		}
		batches[step] = tokens
	}
	return batches, stream.Snapshot(), nil
}

// saveCheckpoint writes the trained weights as a single-file model.safetensors
// and copies config.json + tokenizer.json so the checkpoint reloads through
// densecausal.Load. The trained weights live in m.Weights (the training step
// scatters them back after the final update).
func saveCheckpoint(srcDir, outDir string, m *densecausal.Model) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	meta := map[string]string{"format": "pt", "trainer": "overgo"}
	if err := safetensors.Save(filepath.Join(outDir, "model.safetensors"), m.Weights, m.Shapes, meta); err != nil {
		return err
	}
	for _, name := range []string{"config.json", "tokenizer.json"} {
		if err := copyFile(filepath.Join(srcDir, name), filepath.Join(outDir, name)); err != nil {
			return fmt.Errorf("copy %s: %w", name, err)
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}
