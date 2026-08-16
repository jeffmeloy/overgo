// Package composition runs the composition-viability experiment: graft one
// organ-classified donor component into a frozen target behind a trainable
// bridge, train only the bridge through the shared Muon plan, and render a
// SHIP or REFUSE verdict against the pre-written envelope threshold. The
// verdict contract: SHIP only if every seed's held-out loss lands strictly
// below the unmodified target's (envelope separation at n<=3 per skill.md
// Distribution Discipline); anything else is a measured refusal, never a
// silent pass.
package composition

import (
	"fmt"
	"math"

	"overgo/internal/densecausal"
	"overgo/internal/organ"
)

type Config struct {
	TargetDir  string
	DonorDir   string
	GraftLayer int // -1 selects the target's middle layer
	DonorLayer int // -1 selects the donor's middle layer
	Seeds      []int64
	Steps      int
	BaseLR     float64 // <=0 derives n_params^-1/2
	Momentum   float64
	Train      [][]int
	HeldOut    []int
}

type SeedOutcome struct {
	Seed        int64
	TrainLosses []float64
	HeldOut     float64
}

type Result struct {
	DonorContract organ.Contract
	DonorTensor   string
	GraftLayer    int
	Baseline      float64
	Outcomes      []SeedOutcome
	Ship          bool
	Reason        string
}

// RunViability executes the full experiment. It returns an error only when the
// experiment could not run; a refusal is a successful experiment whose verdict
// is Ship=false with the measured reason.
func RunViability(config Config) (Result, error) {
	if len(config.Seeds) == 0 || config.Steps <= 0 || len(config.Train) == 0 || len(config.HeldOut) < 2 {
		return Result{}, fmt.Errorf("composition: seeds, steps, train batches and a held-out batch are required")
	}
	target, err := densecausal.Load(config.TargetDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: load target: %w", err)
	}
	donor, err := densecausal.Load(config.DonorDir)
	if err != nil {
		return Result{}, fmt.Errorf("composition: load donor: %w", err)
	}
	graftLayer := config.GraftLayer
	if graftLayer < 0 {
		graftLayer = target.Dims.Layers / 2
	}
	names := make([]string, 0, len(donor.Weights))
	for name := range donor.Weights {
		names = append(names, name)
	}
	catalog, err := organ.CompileCatalog(names)
	if err != nil {
		return Result{}, err
	}
	donorLayer := config.DonorLayer
	if donorLayer < 0 {
		donorLayer = catalog.MLPCount() / 2
	}
	component, ok := catalog.MLP(donorLayer)
	if !ok {
		return Result{}, fmt.Errorf("composition: donor MLP index %d outside %d cataloged components", donorLayer, catalog.MLPCount())
	}
	gateName, upName, downName, contract := component.Gate, component.Up, component.Down, component.Contract
	baseline, _, err := target.Loss(config.HeldOut)
	if err != nil {
		return Result{}, fmt.Errorf("composition: baseline held-out loss: %w", err)
	}
	result := Result{
		DonorContract: contract, DonorTensor: gateName, GraftLayer: graftLayer, Baseline: baseline,
	}
	worst := math.Inf(-1)
	for _, seed := range config.Seeds {
		graft, err := densecausal.NewGraft(
			target, graftLayer,
			donor.Weights[gateName], donor.Weights[upName], donor.Weights[downName],
			donor.Dims.Hidden, donor.Dims.Intermediate, seed,
		)
		if err != nil {
			return Result{}, err
		}
		losses, err := target.TrainBridge(graft, config.Train, config.BaseLR, config.Momentum, config.Steps)
		if err != nil {
			return Result{}, fmt.Errorf("composition: seed %d: %w", seed, err)
		}
		heldOut, err := target.GraftLoss(graft, config.HeldOut)
		if err != nil {
			return Result{}, fmt.Errorf("composition: seed %d held-out: %w", seed, err)
		}
		if math.IsNaN(heldOut) || math.IsInf(heldOut, 0) {
			return Result{}, fmt.Errorf("composition: seed %d held-out loss is non-finite", seed)
		}
		result.Outcomes = append(result.Outcomes, SeedOutcome{Seed: seed, TrainLosses: losses, HeldOut: heldOut})
		worst = math.Max(worst, heldOut)
	}
	if worst < baseline {
		result.Ship = true
		result.Reason = fmt.Sprintf(
			"envelope separation: worst seed held-out %.6f < baseline %.6f across %d seeds",
			worst, baseline, len(config.Seeds))
	} else {
		result.Reason = fmt.Sprintf(
			"refusal: worst seed held-out %.6f does not beat baseline %.6f; failure mode: bridge training did not produce held-out gain within %d steps",
			worst, baseline, config.Steps)
	}
	return result, nil
}
