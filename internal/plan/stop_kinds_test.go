package plan

import "testing"

// TestAStopIsTheOwnersOrAnOutsideNeed holds a stop to the owner's request and the two outside needs, a licence the
// owner accepts and a device the machine lacks, each naming what it waits on. The stops of a week that cost 22.5 idle
// hours are refused: a decision between options, a missing file, a download, an irreversible act awaiting a word, and a
// session pacing itself, whatever words their detail holds, as is a manufactured pause and an outside need that names
// nothing. A stop file written with an older kind still reads, as a stop that holds until the owner resumes it.
func TestAStopIsTheOwnersOrAnOutsideNeed(t *testing.T) {
	t.Parallel()
	for reason, want := range map[string][2]string{
		"requested":                              {StopRequested, ""},
		"requested: the owner said stop":         {StopRequested, "the owner said stop"},
		"licence: Gemma's terms on Hugging Face": {stopLicence, "Gemma's terms on Hugging Face"},
		"hardware: a second graphics card":       {stopHardware, "a second graphics card"},
	} {
		code, detail, err := ParseStopReason(reason)
		if err != nil || code != want[0] || detail != want[1] {
			t.Errorf("ParseStopReason(%q) = %q, %q, %v; want %q, %q", reason, code, detail, err, want[0], want[1])
		}
	}
	for _, reason := range []string{
		"external-prereq: outside decision for format-loader/do: the owner chooses retire, a reader seam or a re-keyed store",
		"external-prereq: the model file is missing",
		"external-prereq: download the unquantized model",
		"external-prereq: stage 3 best started with fresh context",
		"irreversible: push the commits",
		"decision: which loader",
		"checkpoint", "should I continue?", "",
		"licence:", "hardware: ",
	} {
		if code, _, err := ParseStopReason(reason); err == nil {
			t.Errorf("ParseStopReason(%q) admitted a %q stop; want it refused", reason, code)
		}
	}
	_, written := stopFixture(t)
	for _, older := range []string{"external-prereq", "irreversible"} {
		written.ReasonCode, written.Detail = older, "an older stop"
		if err := validateWritten(written); err != nil || controlCode(older) != StopRequested {
			t.Errorf("a stop file written as %s reads as %q (%v); want a stop that holds until resumed", older, controlCode(older), err)
		}
	}
}
