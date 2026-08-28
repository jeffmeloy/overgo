// graft-probe: composition viability CLI.
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
	"overgo/internal/clioptions"
	"overgo/internal/composition"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

func main() {
	clioptions.MainNamed("graft-probe", func() error { return run(os.Args[1:], os.Stdout) })
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
	momentum := flags.Float64("mu", trainingprogram.BuiltinOptimizerPolicy().Momentum(), "Muon momentum")
	recordStore := flags.String("record", "", "OvergoDB root: commit the experiment as a generation record with its verdict")
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
	synthesize := flags.String("synthesize", "", "run the full bridge-synthesis pipeline for a committed blocked proposal (artifact ID); emits a typed decision")
	deciderIdentity := flags.String("decider", "", "synthesize: the decider authority evidence artifact ID")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *propose != "" {
		if flags.NArg() != 0 || *proposeTarget == "" || *proposeVerifier == "" || *recordStore == "" {
			return errors.New("usage: graft-probe -propose <retrieval.json> -propose-target <model-id> -propose-verifier <cmd> -record <overgodb>")
		}
		return runPropose(*propose, *proposeTarget, *proposeVerifier, *proposeBlocker, *recordStore, output)
	}
	if *synthesize != "" {
		if flags.NArg() != 0 || *targetDir == "" || *donorDir == "" || *tokensPath == "" || *recordStore == "" || *deciderIdentity == "" {
			return errors.New("usage: graft-probe -synthesize <proposal-id> -decider <evidence-id> -target <dir> -donor <dir> -tokens <ids.json> -record <overgodb>")
		}
		return runSynthesize(*synthesize, *deciderIdentity, *targetDir, *donorDir, *tokensPath, *recordStore, *steps, *window, *lrScale, *momentum, output)
	}
	if *chain {
		if flags.NArg() != 0 || *scorerDir == "" || *drafterDir == "" || *tokensPath == "" || *recordStore == "" {
			return errors.New("usage: graft-probe -chain -scorer <dir> -drafter <dir> -tokens <ids.json> -record <overgodb> [options]")
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
		store, err := overgodb.Open(*recordStore)
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
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ranker, err := composition.TrainProposalRankerFromStore(context.Background(), store)
	if err != nil {
		return err
	}
	proposal, err := composition.NewBridgeProposal(target, candidates, ranker, verifier, blocker)
	if err != nil {
		return err
	}
	rankerContent, err := ranker.Content()
	if err != nil {
		return err
	}
	proposalContent, err := proposal.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch("bridge-proposal/"+proposal.ID.String(),
		[]artifact.Content{rankerContent, proposalContent}, append(ranker.Lineage(), proposal.Lineage()...), nil)
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

// runSynthesize executes the full bridge-synthesis pipeline for one committed
// blocked proposal, emitting a durable typed decision either way.
func runSynthesize(
	proposalText, deciderText, targetDir, donorDir, tokensPath, recordStore string,
	steps, window int, lrScale, momentum float64,
	output io.Writer,
) error {
	var tokens []int
	if err := jsonfile.Decode(tokensPath, &tokens); err != nil {
		return err
	}
	if len(tokens) < 3*window {
		return fmt.Errorf("need at least %d tokens, got %d", 3*window, len(tokens))
	}
	proposalID, err := artifact.ParseID(proposalText)
	if err != nil {
		return err
	}
	deciderID, err := artifact.ParseID(deciderText)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	content, ok, err := artifact.ReadContent(context.Background(), store, proposalID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("proposal %s is not committed", proposalID)
	}
	proposal, err := composition.ParseBridgeProposal(content.Data)
	if err != nil {
		return err
	}
	revision, err := runrecord.HeadCommit(".")
	if err != nil {
		return err
	}
	outcome, err := composition.SynthesizeBridge(store, proposal, composition.Config{
		TargetDir: targetDir, DonorDir: donorDir,
		GraftLayer: -1, DonorLayer: -1,
		Seeds: []int64{7, 11, 13}, Steps: steps,
		BaseLR: 0, LRScale: lrScale, Momentum: momentum,
		Train:   [][]int{tokens[:window], tokens[window : 2*window]},
		HeldOut: tokens[2*window : 3*window],
	}, recipe.Decider{CodeCommit: revision, Derivation: deciderID})
	if err != nil {
		return err
	}
	verdict := "REFUSE"
	if outcome.Result.Ship {
		verdict = "SHIP"
	}
	fmt.Fprintf(output, "synthesis decision committed: %s (%s)\n", outcome.Decision.ID, verdict)
	fmt.Fprintf(output, "generation record: %s\n", outcome.Record.ID)
	fmt.Fprintf(output, "reason: %s\n", outcome.Decision.Reason)
	fmt.Fprintln(output, "honesty: the proposal stays promotion-blocked; this decision is evidence for the experiment plane, never an override")
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
	store, err := overgodb.Open(recordStore)
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
