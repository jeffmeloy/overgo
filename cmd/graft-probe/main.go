// Command graft-probe runs the composition-viability experiment from the
// command line: graft one organ-classified donor MLP into a frozen target
// behind a trainable bridge, train only the bridge with shared Muon, and print
// the SHIP or REFUSE verdict with every measured loss. This is the manual,
// human-reviewed entry point the composition-viability plan row requires.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/composition"
	"overgo/internal/jsonfile"
	"overgo/internal/repodb"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "graft-probe:", err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("graft-probe", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	targetDir := flags.String("target", "", "target model directory (safetensors)")
	donorDir := flags.String("donor", "", "donor model directory (safetensors)")
	tokensPath := flags.String("tokens", "", "JSON array of token IDs; sliced into train and held-out windows")
	graftLayer := flags.Int("layer", -1, "target graft layer (-1 = middle)")
	donorLayer := flags.Int("donor-layer", -1, "donor component layer (-1 = middle)")
	steps := flags.Int("steps", 6, "bridge Muon steps per seed")
	window := flags.Int("window", 64, "tokens per batch window")
	learningRate := flags.Float64("lr", 0, "bridge learning rate (<=0 derives n_params^-1/2)")
	momentum := flags.Float64("mu", 0.9, "Muon momentum")
	recordStore := flags.String("record", "", "RepoDB root: commit the experiment as a generation record with its verdict")
	chain := flags.Bool("chain", false, "run the Tier-0 whole-model chain probe instead of the graft probe")
	scorerDir := flags.String("scorer", "", "chain probe: scorer model directory (safetensors + tokenizer.json)")
	drafterDir := flags.String("drafter", "", "chain probe: drafter model directory (safetensors + tokenizer.json)")
	prefix := flags.Int("chain-prefix", 64, "chain probe: context tokens per window")
	draft := flags.Int("chain-draft", 8, "chain probe: generated gap tokens per arm")
	heldOut := flags.Int("chain-target", 32, "chain probe: held-out tokens scored per window")
	windows := flags.Int("chain-windows", 3, "chain probe: disjoint windows sliced from the token stream")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *chain {
		if flags.NArg() != 0 || *scorerDir == "" || *drafterDir == "" || *tokensPath == "" || *recordStore == "" {
			return errors.New("usage: graft-probe -chain -scorer <dir> -drafter <dir> -tokens <ids.json> -record <repodb> [options]")
		}
		return runChain(*scorerDir, *drafterDir, *tokensPath, *recordStore, *prefix, *draft, *heldOut, *windows, output)
	}
	if flags.NArg() != 0 || *targetDir == "" || *donorDir == "" || *tokensPath == "" {
		return errors.New("usage: graft-probe -target <dir> -donor <dir> -tokens <ids.json> [options]")
	}
	var tokens []int
	if err := jsonfile.Decode(*tokensPath, &tokens); err != nil {
		return err
	}
	if len(tokens) < 3*(*window) {
		return fmt.Errorf("need at least %d tokens for two train windows and one held-out window, got %d", 3*(*window), len(tokens))
	}
	probeConfig := composition.Config{
		TargetDir: *targetDir, DonorDir: *donorDir,
		GraftLayer: *graftLayer, DonorLayer: *donorLayer,
		Seeds: []int64{7, 11, 13}, Steps: *steps,
		BaseLR: *learningRate, Momentum: *momentum,
		Train:   [][]int{tokens[:*window], tokens[*window : 2*(*window)]},
		HeldOut: tokens[2*(*window) : 3*(*window)],
	}
	result, err := composition.RunViability(probeConfig)
	if err != nil {
		return err
	}
	if *recordStore != "" {
		store, err := repodb.Open(*recordStore)
		if err != nil {
			return err
		}
		defer func() { _ = store.Close() }()
		record, err := composition.RecordViability(store, probeConfig, result)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "generation record committed: %s\n", record.ID)
	}
	fmt.Fprintf(output, "donor component: %s (role=%s modality=%s) grafted at target layer %d\n",
		result.DonorTensor, result.DonorContract.Role, result.DonorContract.Modality, result.GraftLayer)
	fmt.Fprintf(output, "baseline held-out CE: %.6f\n", result.Baseline)
	for _, outcome := range result.Outcomes {
		steps := make([]string, len(outcome.TrainLosses))
		for i, loss := range outcome.TrainLosses {
			steps[i] = fmt.Sprintf("%.4f", loss)
		}
		fmt.Fprintf(output, "seed %d: train=[%s] held-out=%.6f\n", outcome.Seed, strings.Join(steps, " "), outcome.HeldOut)
	}
	verdict := "REFUSE"
	if result.Ship {
		verdict = "SHIP"
	}
	fmt.Fprintf(output, "verdict: %s -- %s\n", verdict, result.Reason)
	fmt.Fprintln(output, "honesty: host-reference execution; single corpus slice; verdict binds only this artifact pair, layer, and budget")
	return nil
}

// runChain executes the Tier-0 whole-model chain probe: both arms run through
// the generic workflow runtime as content-addressed recipes, and the verdict
// is committed as a generation record whether it ships or refuses.
func runChain(scorerDir, drafterDir, tokensPath, recordStore string, prefix, draft, target, windows int, output io.Writer) error {
	var tokens []int
	if err := jsonfile.Decode(tokensPath, &tokens); err != nil {
		return err
	}
	span := prefix + draft + target
	if windows < 1 || len(tokens) < windows*span {
		return fmt.Errorf("need %d tokens for %d windows of %d, got %d", windows*span, windows, span, len(tokens))
	}
	sliced := make([][]int, windows)
	for index := range sliced {
		sliced[index] = tokens[index*span : (index+1)*span]
	}
	store, err := repodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	chainConfig := composition.ChainConfig{
		ScorerDir: scorerDir, DrafterDir: drafterDir,
		Prefix: prefix, Draft: draft, Target: target, Windows: sliced,
	}
	result, err := composition.RunChainViability(store, chainConfig)
	if err != nil {
		return err
	}
	record, err := composition.RecordChainViability(store, chainConfig, result)
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "generation record committed: %s\n", record.ID)
	fmt.Fprintf(output, "chain recipe %s baseline recipe %s\n", result.ChainRecipe, result.BaselineRecipe)
	for _, outcome := range result.Outcomes {
		fmt.Fprintf(output, "window %d: baseline CE %.6f chain CE %.6f\n",
			outcome.Window, outcome.BaselineCE, outcome.ChainCE)
	}
	verdict := "REFUSE"
	if result.Ship {
		verdict = "SHIP"
	}
	fmt.Fprintf(output, "verdict: %s -- %s\n", verdict, result.Reason)
	fmt.Fprintln(output, "honesty: host-reference greedy decode; displaced-window CE; verdict binds only this model pair, window protocol, and budget")
	return nil
}
