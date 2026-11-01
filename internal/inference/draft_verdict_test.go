package inference

import (
	"slices"
	"testing"
)

// TestARecordedDraftVerdictRunsNoProbe holds a model's draft judgement to the verdict a judgement on its model, build
// and card recorded: it decodes the probe once in the mode the verdict serves, keeping the first use of the card the
// open's, and judges nothing; without one it probes both modes and refuses drafting where drafted tokens came slower.
func TestARecordedDraftVerdictRunsNoProbe(t *testing.T) {
	t.Parallel()
	rates := map[bool]float64{true: 95, false: 57}
	for _, recorded := range []bool{true, false} {
		var decoded []bool
		probe := func(plain bool) (float64, bool, error) {
			decoded = append(decoded, plain)
			return rates[plain], true, nil
		}
		refused, judged, err := judgeDrafts(&recorded, probe)
		if err != nil || judged || refused != recorded || !slices.Equal(decoded, []bool{recorded}) {
			t.Errorf("a recorded verdict refused=%t decoded %v and gave refused=%t judged=%t (%v); want one decode in its mode and no judgement",
				recorded, decoded, refused, judged, err)
		}
	}
	var decoded []bool
	probe := func(plain bool) (float64, bool, error) {
		decoded = append(decoded, plain)
		return rates[plain], true, nil
	}
	refused, judged, err := judgeDrafts(nil, probe)
	if err != nil || !judged || !refused || !slices.Equal(decoded, []bool{true, false}) {
		t.Errorf("no recorded verdict decoded %v and gave refused=%t judged=%t (%v); want both modes probed and drafting at 57 tok/s against 95 refused",
			decoded, refused, judged, err)
	}
}
