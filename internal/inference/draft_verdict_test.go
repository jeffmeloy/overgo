package inference

import (
	"reflect"
	"strings"
	"testing"
)

// TestTheOpensDraftVerdictIsTheOneDefaultDraftingDecision holds default drafting through a model's own next-token layer
// to one decision, the open's measured verdict (SQA-257): a second rule refused drafting once fewer than one draft in
// ten was kept, though the kept share does not decide the rate (Qwen3.5-4B kept half its drafts and decoded 57 tok/s
// drafted against 95 plainly). The verdict refuses drafting where its tokens come slower and keeps it where they come
// faster, a recorded verdict stands without a probe of both modes, and a runner holds no tally of drafts kept.
func TestTheOpensDraftVerdictIsTheOneDefaultDraftingDecision(t *testing.T) {
	t.Parallel()
	rates := func(plain, drafted float64) func(bool) (float64, bool, error) {
		return func(isPlain bool) (float64, bool, error) {
			if isPlain {
				return plain, false, nil
			}
			return drafted, true, nil
		}
	}
	if refused, judged, err := judgeDrafts(nil, rates(95, 57)); err != nil || !refused || !judged {
		t.Errorf("drafting at 57 tok/s against 95 plainly was refused %v, judged %v (%v); want refused", refused, judged, err)
	}
	if refused, judged, err := judgeDrafts(nil, rates(37, 48)); err != nil || refused || !judged {
		t.Errorf("drafting at 48 tok/s against 37 plainly was refused %v, judged %v (%v); want kept", refused, judged, err)
	}
	probed := 0
	recorded := true
	refused, judged, err := judgeDrafts(&recorded, func(bool) (float64, bool, error) { probed++; return 0, false, nil })
	if err != nil || !refused || judged || probed != 1 {
		t.Errorf("a recorded refusal read refused %v, judged %v after %d probes (%v); want it standing after one decode", refused, judged, probed, err)
	}
	runner := reflect.TypeFor[Runner]()
	for field := range runner.Fields() {
		if name := field.Name; strings.Contains(name, "Kept") || name == "ownDrafted" {
			t.Errorf("the runner holds the kept-share tally %s, a second rule beside the verdict", name)
		}
	}
}
