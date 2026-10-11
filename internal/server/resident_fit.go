package server

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// The plan row's fit rule, after Unsloth Studio's (required times 1.15 plus
// 4 GB), read as free device bytes and GiB and not checked against that source.
const (
	fitMarginPercent = 115
	fitPercent       = 100
	fitReserveBytes  = 4 << 30
)

// errResidentFit refuses a device run that does not fit beside the served model.
var errResidentFit = errors.New("workflow workspace: the device run does not fit beside the served model")

// residentFit is the fit rule over a run's measured peak: what the run needs
// free, whether the device has it free beside the served model, and whether
// it would with the model's bytes released. A need past counting fits neither.
func residentFit(required uint64, memory inference.DeviceMemory) (need uint64, beside, alone bool) {
	scaled, counted := checked.Mul64(required, fitMarginPercent)
	// A hundredth of any count and the reserve sum without overflow.
	need = scaled/fitPercent + fitReserveBytes
	if !counted {
		need = math.MaxUint64
	}
	released := memory.FreeBytes
	for _, owner := range memory.Owners {
		released += owner.Bytes
	}
	return need, need <= memory.FreeBytes, need <= released
}

// measuredTrainingPeak is the largest device peak a training session that
// succeeded recorded under the recipe, zero when none used the device. It reads
// the recipe's own dependents, so its cost follows them, not every served turn.
func measuredTrainingPeak(ctx context.Context, store *overgodb.Store, recipeID artifact.ID) (uint64, error) {
	// A failed read leaves no children, and its error is returned below.
	children, err := store.Children(ctx, recipeID)
	var peak uint64
	for _, edge := range children {
		child, _, err := store.Artifact(ctx, edge.Child)
		if err != nil {
			return 0, err
		}
		// A child the store does not hold has no schema, and is passed by too.
		if child.Schema != runrecord.ServingObservationSchema {
			continue
		}
		observation, err := runrecord.RequireServingObservation(ctx, store, edge.Child)
		if err != nil {
			return 0, err
		}
		if observation.Task == recipe.TaskTraining && observation.Outcome == runrecord.OutcomeSucceeded {
			peak = max(peak, observation.Resources.PeakDeviceBytes)
		}
	}
	return peak, err
}

// admitDeviceRun refuses, with the numbers, a device training run whose
// measured peak does not fit beside the served model; the model stays loaded.
// A run no session measured on the device is admitted: its first run measures it.
func (h *Handler) admitDeviceRun(ctx context.Context, recipeID artifact.ID) error {
	owners, ok := h.generator.(deviceMemoryOwnersAPI)
	if !ok {
		return nil
	}
	memory, err := owners.DeviceMemory(ctx)
	if err != nil {
		// As /health reads it: a model whose device memory cannot be read is
		// no device-resident model to weigh, and the device's own allocation
		// check still stands behind the run.
		return nil
	}
	store, err := h.browseStore(ctx)
	if err != nil {
		return err
	}
	required, err := measuredTrainingPeak(ctx, store, recipeID)
	if err != nil || required == 0 {
		return err
	}
	need, beside, alone := residentFit(required, memory)
	if beside {
		return nil
	}
	verdict := "would not fit either"
	if alone {
		verdict = "would fit"
	}
	return fmt.Errorf("%w: its measured peak of %d bytes needs %d free (the peak times %d/%d plus %d) and %d are free; with the served model released it %s",
		errResidentFit, required, need, fitMarginPercent, fitPercent, fitReserveBytes, memory.FreeBytes, verdict)
}
