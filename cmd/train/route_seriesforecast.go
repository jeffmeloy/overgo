package main

import (
	"context"
	"flag"
	"fmt"

	"overgo/internal/clioptions"
)

// trainSeriesForecast is the seriesforecast route: training under a
// registered forecast objective, whose held-out verdict and trained
// checkpoint it publishes.
func trainSeriesForecast(ctx context.Context, flags *flag.FlagSet, args []string) error {
	modelDir := flags.String("model", "", "forecast model directory (safetensors + config.json)")
	steps := clioptions.IntOverride(flags, "steps", "required observed Muon steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storeRoot := flags.String("store", "overgodb-store", "OvergoDB holding the objective, and receiving the session observation, verdict and checkpoint record")
	objectiveAlias := flags.String("objective", "", "registered forecast objective alias: train on its training membership, one light-curve JSON per dataset file, and judge on its held-out one")
	out := flags.String("out", "", "directory the trained model's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *objectiveAlias == "" || *modelDir == "" || *out == "" || *steps <= 0 {
		return fmt.Errorf("%s: objective, model, out, and positive steps required", flags.Name())
	}
	return runObjective(ctx, *storeRoot, *objectiveAlias, *modelDir, *out, *steps, *maxWall)
}
