package clioptions

import (
	"flag"
	"time"
)

// TrainProbeMain runs the command line every bounded train probe
// shares: a model directory, a dataset path, required observed steps,
// one probe-specific integer bound, and an optional projected-wall
// bound. The probe supplies its bound flag's name and usage plus the
// body that turns the parsed values into training evidence.
func TrainProbeMain(modelUsage, datasetUsage, boundName, boundUsage string, run func(model, dataset string, steps, bound int, maxWall time.Duration) error) {
	model := flag.String("model", "", modelUsage)
	dataset := flag.String("dataset", "", datasetUsage)
	steps := IntOverride(flag.CommandLine, "steps", "required observed Muon steps")
	bound := IntOverride(flag.CommandLine, boundName, boundUsage)
	maxWall := DurationOverride(flag.CommandLine, "max-wall", "optional projected-wall bound")
	flag.Parse()
	Main(func() error {
		return run(*model, *dataset, *steps, *bound, *maxWall)
	})
}
