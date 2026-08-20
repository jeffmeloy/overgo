package optimizer

// Stepper applies one compiled Muon plan on the best available backend.
type Stepper interface {
	Step() error
	Close() error
}

// ObservedStepResult: measured facts of one observed training step — the
// single result shape every observed trainer reports.
type ObservedStepResult struct {
	Loss         float64
	Step         int
	LearningRate float64
	GradientL2   float64
}

// Advance applies one observed optimizer step after the caller has filled
// the gradient buffer: stepper update, step count, derived learning rate.
func Advance(stepper Stepper, config Config, step *int, loss, gradientL2 float64) (ObservedStepResult, error) {
	if err := stepper.Step(); err != nil {
		return ObservedStepResult{}, err
	}
	*step++
	return ObservedStepResult{
		Loss: loss, Step: *step,
		LearningRate: config.LearningRate(*step), GradientL2: gradientL2,
	}, nil
}
