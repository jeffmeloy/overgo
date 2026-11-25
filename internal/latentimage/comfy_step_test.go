package latentimage

import (
	"fmt"
	"testing"
	"time"
)

// comfyStamp is one of a sampler's progress events in a ComfyUI recording: the step it completed and the
// microseconds since the run began.
type comfyStamp struct {
	Value        int   `json:"value"`
	Microseconds int64 `json:"us"`
}

// comfyStepTime is a recorded sampler's time per completed step: from its first stamp to its last, over the steps
// between their numbers. A span of fewer than two stamps, or one whose steps or time do not advance, is refused
// (SQA-477). The time between the stamps had been divided by the count of intervals, so a recording that dropped an
// event read slow. One interior event dropped from fifty read 1.136 s a step against 1.113. A Turbo capture's five
// events over steps 1 to 8 read 129.75 ms against 74.14.
func comfyStepTime(stamps []comfyStamp) (time.Duration, error) {
	if len(stamps) < 2 {
		return 0, fmt.Errorf("the recording stamps %d of the sampler's steps; want two at least to read a step", len(stamps))
	}
	first, last := stamps[0], stamps[len(stamps)-1]
	if last.Value <= first.Value || last.Microseconds <= first.Microseconds {
		return 0, fmt.Errorf("the recording stamps step %d at %d µs and step %d at %d µs; want the later step stamped later",
			first.Value, first.Microseconds, last.Value, last.Microseconds)
	}
	return time.Duration(last.Microseconds-first.Microseconds) * time.Microsecond / time.Duration(last.Value-first.Value), nil
}

// TestARecordedStepIsReadOverItsStepNumbers holds the time a ComfyUI recording gives a step to the steps its stamps
// span (SQA-477). Eight steps a millisecond apart read a millisecond a step, whole or with interior events dropped. One
// stamp is refused, and so is a span whose step does not advance.
func TestARecordedStepIsReadOverItsStepNumbers(t *testing.T) {
	t.Parallel()
	var whole []comfyStamp
	for step := 1; step <= 8; step++ {
		whole = append(whole, comfyStamp{Value: step, Microseconds: int64(step) * 1000})
	}
	dropped := []comfyStamp{whole[0], whole[1], whole[4], whole[7]}
	for name, stamps := range map[string][]comfyStamp{"whole": whole, "with events dropped": dropped} {
		if step, err := comfyStepTime(stamps); err != nil || step != time.Millisecond {
			t.Errorf("the recording %s reads %v a step (%v); want a millisecond", name, step, err)
		}
	}
	for name, stamps := range map[string][]comfyStamp{"one stamp": whole[:1], "one step twice": {whole[2], {Value: 3, Microseconds: 9000}}} {
		if step, err := comfyStepTime(stamps); err == nil {
			t.Errorf("the recording of %s read %v a step; want it refused", name, step)
		}
	}
}
