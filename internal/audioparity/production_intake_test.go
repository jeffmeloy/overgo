package audioparity

import (
	"testing"

	"overgo/internal/testevidence"
)

// TestNativeAudioProductionIntake retains the existing production owners as
// the training baseline while recurrent inference enters the shared session.
// It does not claim recurrent-decoder training or replace the CPU CTC trainer.
func TestNativeAudioProductionIntake(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": native audio intake requires production training, recovery and held-out resource runs")
	}
	t.Run("production-training-and-exact-resume", TestASRProductionTrainingAcceptance)
	t.Run("publication-and-authority-recovery", TestAudioPublicationRecoveryAcceptance)
	t.Run("held-out-quality-and-resource-fitness", TestAudioResourceFitnessAcceptance)
	t.Log("existing CPU CTC production trainer, transformed targets, exact checkpoint/resume, publication refusals and held-out quality/resources executed; no recurrent training, GPU, quality-improvement or promotion claim")
}
