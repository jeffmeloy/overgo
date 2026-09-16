package loop

import "testing"

// A successful worker turn is only an exit event. The next committed-plan
// dispatch executes in the same driver invocation without another prompt.
func TestLoopContinuationAfterCommit(t *testing.T) {
	world := &fakeWorld{steps: []Step{{Item: "first", ID: "do"}, {Item: "second", ID: "do"}}}
	advance := func(w *fakeWorld, _ Step, _ string) string { w.steps = w.steps[1:]; return "" }
	world.behavior = []func(*fakeWorld, Step, string) string{advance, advance}
	result, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 3})
	if err != nil || result.Reason != ReasonPlanComplete || world.call != 2 || world.prompts[1] != "second/do" {
		t.Fatalf("continuation: %+v calls=%d prompts=%v err=%v", result, world.call, world.prompts, err)
	}
}

func TestLoopContinuationHonorsStopAfterCommit(t *testing.T) {
	world := &fakeWorld{steps: []Step{{Item: "first", ID: "do"}, {Item: "second", ID: "do"}}}
	world.behavior = []func(*fakeWorld, Step, string) string{func(w *fakeWorld, _ Step, _ string) string {
		w.steps = w.steps[1:]
		w.paused = true
		return ""
	}}
	result, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 3})
	if err != nil || result.Reason != ReasonPaused || world.call != 1 {
		t.Fatalf("stop: %+v calls=%d err=%v", result, world.call, err)
	}
}
