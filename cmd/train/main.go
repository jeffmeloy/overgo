// Command train runs Muon training on a dense causal-LM
// safetensors artifact and writes a trained checkpoint. It is the production
// caller for the densecausal training lane: it loads a model + tokenizer,
// encodes a UTF-8 text file into one token sequence, runs the training step on
// the GPU when available (host fallback), and persists the result via
// safetensors.Save. Without a caller like this the lane is exercised only by
// tests -- present but inert.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
	"overgo/internal/safetensors"
)

func main() {
	clioptions.MainNamed("train", run)
}

func run() error {
	modelDir := flag.String("model", "", "model directory (safetensors + config.json + tokenizer.json)")
	textPath := flag.String("text", "", "UTF-8 training text, encoded to one token sequence")
	outDir := flag.String("out", "", "output directory for the trained checkpoint")
	steps := flag.Int("steps", 1, "number of Muon update steps")
	maxSeq := flag.Int("seq", 512, "cap the token sequence to this length (attention is O(seq^2)); <=0 keeps all")
	baseLR := flag.Float64("lr", 0, "base learning rate; <=0 derives n_params^-1/2")
	mu := flag.Float64("momentum", 0.9, "Muon momentum")
	host := flag.Bool("host", false, "force the host path even when CUDA is available")
	freezeLexical := flag.Bool("freeze-lexical", false, "freeze tied embedding/head; requires CUDA resident training")
	flag.Parse()

	if *modelDir == "" || *textPath == "" || *outDir == "" {
		flag.Usage()
		return fmt.Errorf("-model, -text, and -out are required")
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
	raw, err := os.ReadFile(*textPath)
	if err != nil {
		return fmt.Errorf("read text: %w", err)
	}
	tokens, err := tok.Encode(string(raw))
	if err != nil {
		return fmt.Errorf("encode text: %w", err)
	}
	if len(tokens) < 2 {
		return fmt.Errorf("need >= 2 tokens to form a next-token target, got %d", len(tokens))
	}
	if *maxSeq > 0 && len(tokens) > *maxSeq {
		tokens = tokens[:*maxSeq]
	}

	if *host && *freezeLexical {
		return fmt.Errorf("-host and -freeze-lexical are mutually exclusive")
	}
	traj, backend, err := runTraining(model, tokens, *steps, *baseLR, *mu, !*host, *freezeLexical)
	if err != nil {
		return fmt.Errorf("train: %w", err)
	}
	if err := saveCheckpoint(*modelDir, *outDir, model); err != nil {
		return fmt.Errorf("save checkpoint: %w", err)
	}

	fmt.Printf("backend=%s tokens=%d steps=%d lr=%s momentum=%g freeze_lexical=%v\n", backend, len(tokens), *steps, lrLabel(*baseLR), *mu, *freezeLexical)
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
	tokens []int,
	steps int,
	baseLR, mu float64,
) ([]float64, string, error) {
	trajectory, err := m.Train(tokens, steps, baseLR, mu)
	return trajectory, "host", err
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
