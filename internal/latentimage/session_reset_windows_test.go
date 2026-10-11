//go:build windows

package latentimage

import (
	"reflect"
	"strings"
	"testing"
)

// TestSessionRefusesAnotherTimestepScale holds a resident picture session to
// the training timesteps it was loaded for. A session encodes each step's
// timestep at the scale of that count, fixed when its programs are compiled,
// so a request under another count would be drawn with the new schedule and
// the old encoding, and would not be the picture a session loaded for it
// draws.
//
// A session loaded for 500 and given a request for 1000 refuses it as
// needing a session of its own, the refusal whoever holds the session answers
// by loading one, and refuses before anything of it has changed: its request,
// its schedule and its readiness are as they were. A request that leaves the
// count unnamed takes the profile's, which is the session's, and one that
// names the same count is the same request: both are taken, with the new
// seed.
func TestSessionRefusesAnotherTimestepScale(t *testing.T) {
	const loadedFor, another = 500, 1000
	loaded := Request{Prompt: "a fixture", Width: 64, Height: 64, Steps: 4, Seed: 1, DynamicShiftMu: 1, NumTrainTimesteps: loadedFor}
	schedule, err := CompileFlowSchedule(loaded.Steps, loaded.NumTrainTimesteps, loaded.DynamicShiftMu)
	if err != nil {
		t.Fatal(err)
	}
	var profile Profile
	profile.Sampling = samplingPolicy{DefaultSteps: loaded.Steps, TrainTimesteps: loadedFor}
	generator := &Generator{
		request: loaded, profile: profile, spec: &Spec{}, schedule: schedule, completed: true,
		pipeline: &ResidentImagePipeline{Denoiser: &DenoiserProgram{GH: 4, GW: 4}},
	}
	rescaled := loaded
	rescaled.NumTrainTimesteps = another
	if err := generator.Reset(t.Context(), rescaled); err == nil || !strings.Contains(err.Error(), "requires another resident session") {
		t.Fatalf("a session loaded for %d training timesteps, given a request for %d = %v, want it refused as needing a session of its own", loadedFor, another, err)
	}
	if generator.request != loaded || !reflect.DeepEqual(generator.schedule, schedule) || !generator.completed {
		t.Fatalf("the refused request left the session with request %+v, ready %t; want it as it was", generator.request, generator.completed)
	}
	for name, count := range map[string]int{"leaves the count unnamed": 0, "names the same count": loadedFor} {
		next := loaded
		next.Seed, next.NumTrainTimesteps = loaded.Seed+1, count
		generator.completed = true
		if err := generator.Reset(t.Context(), next); err != nil || generator.request.NumTrainTimesteps != loadedFor || generator.request.Seed != next.Seed {
			t.Fatalf("a request that %s = %v, the session now for %d under seed %d; want it taken for %d under the new seed", name, err, generator.request.NumTrainTimesteps, generator.request.Seed, loadedFor)
		}
	}
}
