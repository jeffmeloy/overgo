package webuilane

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
)

// TestALaneReadsATestWhoseDataIsBeingRemadeAsDeclared holds the lane's verdict on a test declared while its data is
// re-created (speech-journey): a journey whose capture speech-captures remakes, declared of that kind, is read as
// declared and credited nothing, so the lane stands on its other journeys; the same journey declared inapplicable, or
// skipped with no kind, still proves nothing and refuses the lane, as does a run where the declared journey is the
// only one. The verdict reads the report go test's own stream gives, through the evidence parser.
func TestALaneReadsATestWhoseDataIsBeingRemadeAsDeclared(t *testing.T) {
	t.Parallel()
	const pkg = "overgo/internal/audioparity"
	stream := func(kind string, other bool) string {
		var events []string
		add := func(fields map[string]string) {
			events = append(events, string(testcheck.Must(json.Marshal(fields))(t)))
		}
		add(map[string]string{"Action": "run", "Package": pkg, "Test": "TestModelJourneyNativeASRMicrophone"})
		if kind != "" {
			add(map[string]string{"Action": "attr", "Package": pkg, "Test": "TestModelJourneyNativeASRMicrophone", "Key": testskip.Key, "Value": kind})
		}
		add(map[string]string{"Action": "output", "Package": pkg, "Test": "TestModelJourneyNativeASRMicrophone", "Output": "    skipped\n"})
		add(map[string]string{"Action": "skip", "Package": pkg, "Test": "TestModelJourneyNativeASRMicrophone"})
		if other {
			add(map[string]string{"Action": "run", "Package": pkg, "Test": "TestModelJourneyCPUASR"})
			add(map[string]string{"Action": "pass", "Package": pkg, "Test": "TestModelJourneyCPUASR"})
		}
		add(map[string]string{"Action": "pass", "Package": pkg})
		return strings.Join(events, "\n") + "\n"
	}
	verdict := func(kind string, other bool) (testevidence.GoTestReport, error) {
		report := testcheck.Must(testevidence.GoTestJSONReport(stream(kind, other)))(t)
		return report, LaneVerdict(report, nil, nil)
	}
	report, err := verdict(testskip.KindRemade, true)
	if err != nil || !slices.Equal(report.Remade, []string{pkg + ": TestModelJourneyNativeASRMicrophone"}) {
		t.Errorf("a journey declared while its data is remade beside a passing journey gives %v, remade %v; want the lane to stand, the journey declared", err, report.Remade)
	}
	for _, kind := range []string{testskip.KindInapplicable, ""} {
		if _, err := verdict(kind, true); err == nil || !strings.Contains(err.Error(), "was skipped, so it proves nothing") {
			t.Errorf("a journey skipped as %q gives %v; want the lane refused", kind, err)
		}
	}
	if _, err := verdict(testskip.KindRemade, false); err == nil {
		t.Error("a lane whose only journey is declared while its data is remade stands; want no passing test refused")
	}
}
