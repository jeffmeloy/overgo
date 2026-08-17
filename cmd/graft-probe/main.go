// Command graft-probe runs the composition-viability experiment from the
// command line: graft one organ-classified donor MLP into a frozen target
// behind a trainable bridge, train only the bridge with shared Muon, and print
// the SHIP or REFUSE verdict with every measured loss. This is the manual,
// human-reviewed entry point the composition-viability plan row requires.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/artifact"
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
	lrScale := flags.Float64("lr-scale", 0, "scale the derived learning rate (retrial protocol; effective only when -lr is unset)")
	momentum := flags.Float64("mu", 0.9, "Muon momentum")
	recordStore := flags.String("record", "", "RepoDB root: commit the experiment as a generation record with its verdict")
	chain := flags.Bool("chain", false, "run the Tier-0 whole-model chain probe instead of the graft probe")
	propose := flags.String("propose", "", "convert a similarity-retrieval JSON response into a blocked bridge proposal (path to the response)")
	proposeTarget := flags.String("propose-target", "", "propose: target model artifact ID")
	proposeVerifier := flags.String("propose-verifier", "", "propose: the failable verifier any experiment must run")
	proposeBlocker := flags.String("propose-blocker", "no recorded experiment evidence; promotion requires the experiment plane's admission", "propose: why the candidates are blocked")
	scorerDir := flags.String("scorer", "", "chain probe: scorer model directory (safetensors + tokenizer.json)")
	drafterDir := flags.String("drafter", "", "chain probe: drafter model directory (safetensors + tokenizer.json)")
	prefix := flags.Int("chain-prefix", 64, "chain probe: context tokens per window")
	draft := flags.Int("chain-draft", 8, "chain probe: generated gap tokens per arm")
	heldOut := flags.Int("chain-target", 32, "chain probe: held-out tokens scored per window")
	windows := flags.Int("chain-windows", 3, "chain probe: disjoint windows sliced from the token stream")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *propose != "" {
		if flags.NArg() != 0 || *proposeTarget == "" || *proposeVerifier == "" || *recordStore == "" {
			return errors.New("usage: graft-probe -propose <retrieval.json> -propose-target <model-id> -propose-verifier <cmd> -record <repodb>")
		}
		return runPropose(*propose, *proposeTarget, *proposeVerifier, *proposeBlocker, *recordStore, output)
	}
	inject := flags.Bool("inject", false, "run the residual-injection connector probe (scorer/drafter as in -chain; tokens sliced into sliding train windows and trailing held-out windows)")
	if *inject {
		if flags.NArg() != 0 || *scorerDir == "" || *drafterDir == "" || *tokensPath == "" {
			return errors.New("usage: graft-probe -inject -scorer <dir> -drafter <dir> -tokens <ids.json> [options]")
		}
		return runInject(*scorerDir, *drafterDir, *tokensPath, *prefix, *draft, *heldOut, output)
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
		BaseLR: *learningRate, LRScale: *lrScale, Momentum: *momentum,
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

// runPropose converts one similarity-retrieval response into a committed
// bridge proposal: advisory by construction, promotion-blocked, carrying the
// verifier any future experiment must run. It never authorizes anything.
func runPropose(retrievalPath, targetModel, verifier, blocker, recordStore string, output io.Writer) error {
	var retrieval struct {
		Neighbors []struct {
			Model    string  `json:"model"`
			Name     string  `json:"name"`
			Distance float64 `json:"distance"`
		} `json:"neighbors"`
	}
	if err := jsonfile.Decode(retrievalPath, &retrieval); err != nil {
		return err
	}
	target, err := artifact.ParseID(targetModel)
	if err != nil {
		return err
	}
	candidates := make([]composition.BridgeCandidate, 0, len(retrieval.Neighbors))
	for _, neighbor := range retrieval.Neighbors {
		donor, err := artifact.ParseID(neighbor.Model)
		if err != nil {
			return fmt.Errorf("neighbor %q: %w", neighbor.Name, err)
		}
		if donor == target {
			continue // within-model neighbors are not composition candidates
		}
		candidates = append(candidates, composition.BridgeCandidate{
			Donor: donor, Component: neighbor.Name, Distance: neighbor.Distance,
		})
	}
	proposal, err := composition.NewBridgeProposal(target, candidates, verifier, blocker)
	if err != nil {
		return err
	}
	store, err := repodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	batch, err := proposal.Batch("bridge-proposal/" + proposal.ID.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return err
	}
	fmt.Fprintf(output, "bridge proposal committed: %s\n", proposal.ID)
	fmt.Fprintf(output, "state: %s -- %s\n", proposal.State, proposal.Blocker)
	fmt.Fprintf(output, "candidates: %d; required verifier: %s\n", len(proposal.Candidates), proposal.RequiredVerifier)
	fmt.Fprintln(output, "honesty: advisory by construction; this document cannot authorize work")
	return nil
}

// runInject executes the residual-injection connector probe: a ridge-fit
// linear connector trained by feature matching against measured context gaps,
// evaluated on trailing held-out windows. Refusals print exactly as loudly as
// ships.
func runInject(scorerDir, drafterDir, tokensPath string, prefix, gap, target int, output io.Writer) error {
	var tokens []int
	if err := jsonfile.Decode(tokensPath, &tokens); err != nil {
		return err
	}
	span := prefix + gap + target
	if len(tokens) < 3*span {
		return fmt.Errorf("need at least %d tokens, got %d", 3*span, len(tokens))
	}
	heldOutStart := len(tokens) - 2*span
	var train [][]int
	for start := 0; start+span <= heldOutStart; start += 16 {
		train = append(train, tokens[start:start+span])
	}
	result, err := composition.RunInjectionViability(composition.InjectionConfig{
		ScorerDir: scorerDir, DrafterDir: drafterDir,
		Prefix: prefix, Gap: gap, Target: target, Alpha: 0.25, Ridge: 1e-3,
		Train: train,
		HeldOut: [][]int{
			tokens[heldOutStart : heldOutStart+span],
			tokens[heldOutStart+span : heldOutStart+2*span],
		},
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "scorer layer %d drafter layer %d train windows %d fit residual %.6f\n",
		result.ScorerLayer, result.DrafterLayer, result.TrainWindows, result.FitResidual)
	for _, outcome := range result.Outcomes {
		fmt.Fprintf(output, "held-out %d: baseline CE %.6f injected CE %.6f\n",
			outcome.Window, outcome.BaselineCE, outcome.InjectedCE)
	}
	verdict := "REFUSE"
	if result.Ship {
		verdict = "SHIP"
	}
	fmt.Fprintf(output, "verdict: %s -- %s\n", verdict, result.Reason)
	fmt.Fprintln(output, "honesty: host-reference; feature-matching connector, never task CE; verdict binds only this model pair, protocol, and budget")
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
