package trainingworkflow

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/testutil"
)

// TestRewardAdoptionRequiresDeclaredComparisons holds the pilot's adoption to
// the comparisons its acceptance named. The reward arm is adopted when every
// one of them was charged under that acceptance and answered a gain, and for
// no other record: a comparison that was not made, one with no charge, one
// charged under another acceptance, a loss, and not enough evidence each
// leave the base in place and are named as the reason; and an acceptance that
// names no comparison adopts nothing.
func TestRewardAdoptionRequiresDeclaredComparisons(t *testing.T) {
	acceptance := evaluation.InstructionAcceptance{ID: testutil.ArtifactID(t, artifact.KindEvidence, "pilot acceptance"), Comparisons: pilotComparisons}
	answered := func(verdicts ...string) []PilotComparison {
		comparisons := make([]PilotComparison, len(verdicts))
		for index, verdict := range verdicts {
			comparisons[index] = PilotComparison{Name: pilotComparisons[index].Name, Confirmation: evaluation.PoolConfirmation{
				Acceptance: acceptance.ID, Charge: testutil.ArtifactID(t, artifact.KindEvidence, "charge "+pilotComparisons[index].Name), Verdict: verdict,
			}}
		}
		return comparisons
	}
	if adopted, reason := pilotAdopted(acceptance, answered("gain", "gain")); !adopted || reason != "" {
		t.Fatalf("two charged gains = adopted %t (%s), want the reward arm adopted", adopted, reason)
	}
	uncharged, elsewhere := answered("gain", "gain"), answered("gain", "gain")
	uncharged[1].Confirmation.Charge, elsewhere[0].Confirmation.Acceptance = artifact.ID{}, testutil.ArtifactID(t, artifact.KindEvidence, "another acceptance")
	for name, seeded := range map[string]struct {
		comparisons []PilotComparison
		reason      string
	}{
		"a comparison that was not made":        {answered("gain"), "was not made"},
		"a comparison with no charge":           {uncharged, "was not charged"},
		"a comparison under another acceptance": {elsewhere, "was not charged"},
		"a loss":                                {answered("gain", "loss"), "answered loss"},
		"not enough evidence":                   {answered("not enough evidence", "gain"), "answered not enough evidence"},
		"no comparison at all":                  {nil, "was not made"},
	} {
		if adopted, reason := pilotAdopted(acceptance, seeded.comparisons); adopted || !strings.Contains(reason, seeded.reason) {
			t.Errorf("%s = adopted %t (%q), want the base kept because it %s", name, adopted, reason, seeded.reason)
		}
	}
	if adopted, _ := pilotAdopted(evaluation.InstructionAcceptance{ID: acceptance.ID}, nil); adopted {
		t.Error("an acceptance that names no comparison adopted an arm")
	}
}
