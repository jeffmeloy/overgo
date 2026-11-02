package mosstrain

import (
	"testing"
)

// TestATrainingDrawStandsOnTheSourceGrid: the draw sigma = t/N is the grid MOSS-Image-SFT's training_shift of 1 builds, the
// value its scheduler_config.json holds, at its 1000 timesteps. Another shift would train off the grid inference reads, and
// is refused.
func TestATrainingDrawStandsOnTheSourceGrid(t *testing.T) {
	t.Parallel()
	const timesteps, sourceShift, otherShift = 1000, 1.0, 3.0
	if err := CheckTrainingGrid(timesteps, sourceShift); err != nil {
		t.Fatalf("the source's grid: %v", err)
	}
	if err := CheckTrainingGrid(timesteps, otherShift); err == nil {
		t.Fatal("a run on another shift's grid passes")
	}
}
