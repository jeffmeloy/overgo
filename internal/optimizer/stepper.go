package optimizer

// Stepper applies one compiled Muon plan on the best available backend.
type Stepper interface {
	Step() error
	Close() error
}
