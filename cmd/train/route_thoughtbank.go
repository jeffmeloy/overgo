package main

import (
	"context"
	"flag"
	"fmt"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/hfbpe"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/thoughtbank"
	"overgo/internal/trainingprogram"
)

// thoughtbankCheckpoint is the self-describing PyTorch checkpoint a
// thoughtbank model directory holds beside its tokenizer.json.
const thoughtbankCheckpoint = "model.pt"

// trainThoughtBank is the thoughtbank route: under a supervised session it
// trains one block's fast-weight bank read of a thoughtbank checkpoint on
// the leading tokens of a text dataset, the read mapping each token's
// embedding to the next token's through a memory slot holding the leading
// width of the embedding of the token after them, refuses a run whose loss
// does not descend, and publishes the trained bank as a checkpoint.
func trainThoughtBank(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	model := flags.String("model", "", "thoughtbank model directory (model.pt + tokenizer.json)")
	datasetPath := flags.String("dataset", "", "UTF-8 training dataset")
	layer := flags.Int("layer", 0, "the block whose bank trains")
	rows := clioptions.IntOverride(flags, "rows", "required token rows the bank reads")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB for the session observation, recipe authority and checkpoint record")
	out := flags.String("out", "", "directory the trained bank's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	name := flags.Name()
	if *model == "" || *datasetPath == "" || *out == "" || *steps <= 0 || *rows <= 0 || *layer < 0 {
		return fmt.Errorf("%s: model, dataset, out, positive steps and rows, and a nonnegative layer required", name)
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	directory := roots.ResolveModelPath(*model)
	checkpointPath := filepath.Join(directory, thoughtbankCheckpoint)
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	session, err := openSession(ctx, store, checkpointPath, *datasetPath)
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	weights, _, err := thoughtbank.LoadCheckpoint(checkpointPath)
	if err != nil {
		return err
	}
	if *layer >= len(weights.Blocks) {
		return fmt.Errorf("%s: layer %d outside the checkpoint's %d blocks", name, *layer, len(weights.Blocks))
	}
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		return err
	}
	tokens, err := tokenizer.Encode(string(session.data))
	if err != nil {
		return err
	}
	// The rows, the last row's successor, and the memory's token.
	need := *rows + 1 + 1
	if len(tokens) < need {
		return fmt.Errorf("%s: dataset yields %d tokens, need %d", name, len(tokens), need)
	}
	for _, token := range tokens[:need] {
		if token < 0 || token >= weights.VocabSize {
			return fmt.Errorf("%s: token %d outside the checkpoint's vocabulary of %d", name, token, weights.VocabSize)
		}
	}
	d, width := weights.DModel, weights.MemDim
	if width > d {
		return fmt.Errorf("%s: memory width %d exceeds the embedding width %d", name, width, d)
	}
	embedding := func(token int) []float32 { return weights.Embed[token*d : (token+1)*d] }
	example := thoughtbank.BankTrainingExample{
		Input: make([]float32, 0, *rows*d), Target: make([]float32, 0, *rows*d),
		Memory: embedding(tokens[*rows+1])[:width], Rows: *rows, Slots: 1,
	}
	for row := range *rows {
		example.Input = append(example.Input, embedding(tokens[row])...)
		example.Target = append(example.Target, embedding(tokens[row+1])...)
	}
	bank := weights.Blocks[*layer].Bank
	parameters := len(bank.FWA) + len(bank.FWB) + len(bank.FWO) + len(bank.NormWeight)
	plan, err := probeSession(trainingprogram.ObjectiveTokenPrediction, *steps, parameters, *maxWall)
	if err != nil {
		return err
	}
	config := plan.Optimizer()
	trainer, err := bank.NewTrainer(config)
	if err != nil {
		return err
	}
	defer trainer.Close()
	fmt.Printf("trainable parameters=%d layer=%d rows=%d derived_lr=%.4g derived_momentum=%.4f\n",
		parameters, *layer, *rows, config.BaseLearningRate, config.Momentum)
	return boundedRun{name: name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			result, err := trainer.Step(example)
			return optimizer.ObservedStepResult{
				Loss: result.Loss, Step: step + 1, LearningRate: config.LearningRate(step + 1), GradientL2: result.GradientL2,
			}, err
		},
		loss: func() (float64, error) { return trainer.Loss(example) },
		// Every update trains on the one example the dataset's leading
		// tokens make.
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, *out, trainer, plan.Updates())
		},
	}.train(ctx)
}
