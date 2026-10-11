package densecausaltrain

import (
	"math"
	"testing"
)

// TestRewardReferenceTakesTheHostRewardTrainersSteps holds the reference the
// device's reward steps are compared with (rewardReference) to the host's
// reward trainer, while the host has one. Both are host arithmetic, so the
// graphics card has no part in it: over three groups whose answers have an
// order, of unequal lengths, the reference's losses and the weights it ends
// at are the trainer's, within what two orders of the same single-precision
// sums differ by after the optimizer has stepped on them, and the weights
// have moved by far more. A reference whose weights kept the sign the order
// gives them steps the other way and stands far outside that.
func TestRewardReferenceTakesTheHostRewardTrainersSteps(t *testing.T) {
	// What the two were measured to differ by: 8.8e-10, 1.9e-7 and 2.4e-6 in
	// the three groups' losses, growing as the weights part, and 5.0e-5 in the
	// weights after the three, which had moved 5.3e-2. The weights differ by
	// more than single precision does because the optimizer steps along a
	// direction it makes from the gradient and not along the gradient, and a
	// last-place change in a gradient turns that direction: the reference
	// taken with its weights rounded to single precision ended 2.7e-5 from
	// itself. The bounds are four and eight times what was measured.
	const momentum, lossBound, weightBound = 0.9, 1e-5, 4e-4
	ordered := orderedRewardGroups(t)
	host, expected := tinyMuonModel(t), tinyMuonModel(t)
	observed, _, err := host.TrainGRPOGroupsResume(ordered, derivedTestLearningRate(t, host), momentum, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	losses := rewardReference(t, expected, ordered, momentum)
	if len(observed) != len(ordered) || len(losses) != len(ordered) {
		t.Fatalf("%d reward groups gave %d steps by the host's trainer and %d by the reference", len(ordered), len(observed), len(losses))
	}
	for step, observation := range observed {
		if math.Abs(observation.Loss-losses[step]) > lossBound {
			t.Errorf("reward group %d: the reference's loss %.9f, the host trainer's %.9f; want them within %.0e", step, losses[step], observation.Loss, lossBound)
		}
	}
	if furthest, moved := apart(host, expected), apart(expected, tinyMuonModel(t)); furthest > weightBound || moved <= weightBound {
		t.Errorf("after %d reward groups the reference's weights stand %.2e from the host trainer's and %.2e from where they began; want them within %.0e of the trainer's and moved by more", len(ordered), furthest, moved, weightBound)
	}
}
