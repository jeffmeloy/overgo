package main

import (
	"errors"
	"strings"
	"testing"
)

// TestInventoryJudgesMediaEvidenceByTheChain holds the media currency rule to
// the acceptance package's go test run at this tree: a surface's evidence is
// current only when the run passed and every acceptance the surface names
// among its verdicts passed with at least one named. A failed run, a skipped
// acceptance or a run naming none of the surface's acceptances each refuse
// with the reason; a surface no acceptance judges is not judged at all.
func TestInventoryJudgesMediaEvidenceByTheChain(t *testing.T) {
	t.Parallel()
	passing := testReport{Passed: true, Tests: map[string]string{
		"TestImageVideoLifecycleAcceptance": "pass", "TestMediaWorkflowruntimeReconciliation": "pass",
		"TestAcceptedRxBrainVQA": "pass", "TestAcceptedSpecializedTaskEvidence": "pass", "TestImageVideoReportContract": "pass",
	}}
	for _, surface := range []SurfaceID{"image", "video", "vqa", "specialized"} {
		if err := mediaCurrent(passing, surface); err != nil {
			t.Errorf("%s judged not current by a passing receipt: %v", surface, err)
		}
	}
	if err := mediaCurrent(passing, "training"); !errors.Is(err, errCellNotJudged) {
		t.Errorf("an unjudged surface was judged: %v", err)
	}
	// Speech is judged by its own package's receipt: the speech synthesizer's
	// goldens over the model's artifact, not the media acceptances.
	if err := mediaCurrent(passing, "speech"); err == nil || errors.Is(err, errCellNotJudged) {
		t.Errorf("the media receipt judged speech evidence: %v", err)
	}
	goldens := testReport{Passed: true, Tests: map[string]string{"TestSpeakE2EProducesReferenceAudio": "pass", "TestGenerationGolden": "pass", "TestSynthWall": "pass"}}
	if err := mediaCurrent(goldens, "speech"); err != nil {
		t.Errorf("a passing golden receipt refused speech evidence: %v", err)
	}
	if surfacePackages["speech"] != speechAcceptancePackage || surfacePackages["image"] != mediaAcceptancePackage {
		t.Errorf("surface packages = %v", surfacePackages)
	}
	// A skip in a receipt is a declared exclusion: a test that stated why it
	// cannot apply at this tree. It neither judges nor refuses the surface;
	// the surface is current when another named acceptance passed and not
	// when every named acceptance was excluded.
	excluded := passing
	excluded.Tests = map[string]string{"TestImageVideoResourceProtocolAcceptance": "skip", "TestImageVideoWanAcceptance": "pass"}
	if err := mediaCurrent(excluded, "image"); err != nil {
		t.Errorf("a declared exclusion beside a passing acceptance refused image evidence: %v", err)
	}
	for name, report := range map[string]testReport{
		"failed":        {Passed: false},
		"only excluded": {Passed: true, Tests: map[string]string{"TestImageVideoResourceProtocolAcceptance": "skip"}},
		"unnamed":       {Passed: true, Tests: map[string]string{"TestImageVideoReportContract": "pass"}},
	} {
		err := mediaCurrent(report, "image")
		if err == nil || errors.Is(err, errCellNotJudged) {
			t.Errorf("%s receipt judged image evidence current: %v", name, err)
		}
	}
	if err := mediaCurrent(testReport{Passed: false}, "video"); err == nil || !strings.Contains(err.Error(), "failed") {
		t.Errorf("a failed run's refusal does not say so: %v", err)
	}
}
