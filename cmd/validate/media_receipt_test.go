package main

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testevidence"
	"overgo/internal/testskip"
)

// acceptanceReceipt is a tiny go test -json run read by testevidence, as
// packageTests reads one: each outcome is "package Test action", where the
// action inapplicable records the kind testskip.NotApplicable declares before
// its skip, and passed is go test's exit.
func acceptanceReceipt(t *testing.T, passed bool, outcomes ...string) testReport {
	t.Helper()
	var stream strings.Builder
	event := func(fields map[string]string) {
		data, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		stream.Write(append(data, '\n'))
	}
	var packages []string
	failed := map[string]bool{}
	for _, outcome := range outcomes {
		fields := strings.Fields(outcome)
		pkg, test, action := fields[0], fields[1], fields[2]
		if !slices.Contains(packages, pkg) {
			packages = append(packages, pkg)
			event(map[string]string{"Action": "start", "Package": pkg})
		}
		event(map[string]string{"Action": "run", "Package": pkg, "Test": test})
		if action == "inapplicable" {
			event(map[string]string{"Action": "attr", "Package": pkg, "Test": test, "Key": testskip.Key, "Value": testskip.KindInapplicable})
			action = "skip"
		}
		failed[pkg] = failed[pkg] || action == "fail"
		event(map[string]string{"Action": action, "Package": pkg, "Test": test})
	}
	for _, pkg := range packages {
		event(map[string]string{"Action": map[bool]string{false: "pass", true: "fail"}[failed[pkg]], "Package": pkg})
	}
	report, err := testevidence.GoTestJSONReport(stream.String())
	if err != nil {
		t.Fatal(err)
	}
	return testReport{Passed: passed, Report: report}
}

// TestInventoryJudgesMediaEvidenceByTheChain holds the media currency rule to
// the acceptance package's go test run at this tree, read by testevidence: a
// surface's evidence is current only when the run passed, its package's run
// completed, and an acceptance the surface names passed in that package. A
// failed run, a skip that declared no inapplicability, a run naming none of
// the surface's acceptances, or one whose acceptance passed only in another
// package each refuse with the reason; acceptances that only declared
// themselves inapplicable refuse apart from failures; a surface no acceptance
// judges is not judged at all.
func TestInventoryJudgesMediaEvidenceByTheChain(t *testing.T) {
	t.Parallel()
	const media, speech = "overgo/cmd/validate", "overgo/internal/speechsynth"
	passing := acceptanceReceipt(t, true,
		media+" TestImageVideoLifecycleAcceptance pass", media+" TestMediaWorkflowruntimeReconciliation pass",
		media+" TestAcceptedRxBrainVQA pass", media+" TestAcceptedSpecializedTaskEvidence pass", media+" TestImageVideoReportContract pass")
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
	goldens := acceptanceReceipt(t, true, speech+" TestSpeakE2EProducesReferenceAudio pass", speech+" TestGenerationGolden pass", speech+" TestSynthWall pass")
	if err := mediaCurrent(goldens, "speech"); err != nil {
		t.Errorf("a passing golden receipt refused speech evidence: %v", err)
	}
	if surfacePackages["speech"] != speechAcceptancePackage || surfacePackages["image"] != mediaAcceptancePackage {
		t.Errorf("surface packages = %v", surfacePackages)
	}
	// A declared exclusion -- a test that stated why it cannot apply at this
	// tree -- neither judges the surface nor refuses it: the surface is
	// current when another named acceptance passed beside it.
	excluded := acceptanceReceipt(t, true, media+" TestImageVideoResourceProtocolAcceptance inapplicable", media+" TestImageVideoWanAcceptance pass")
	if err := mediaCurrent(excluded, "image"); err != nil {
		t.Errorf("a declared exclusion beside a passing acceptance refused image evidence: %v", err)
	}
	for name, report := range map[string]testReport{
		"failed":                {Passed: false},
		"failed acceptance":     acceptanceReceipt(t, false, media+" TestImageVideoWanAcceptance fail", media+" TestImageVideoLifecycleAcceptance pass"),
		"unnamed":               acceptanceReceipt(t, true, media+" TestImageVideoReportContract pass"),
		"undeclared skip":       acceptanceReceipt(t, true, media+" TestImageVideoResourceProtocolAcceptance skip", media+" TestImageVideoWanAcceptance pass"),
		"another package's run": acceptanceReceipt(t, true, "overgo/internal/other TestImageVideoWanAcceptance pass", media+" TestImageVideoReportContract pass"),
		"only excluded":         acceptanceReceipt(t, true, media+" TestImageVideoResourceProtocolAcceptance inapplicable"),
	} {
		err := mediaCurrent(report, "image")
		if err == nil || errors.Is(err, errCellNotJudged) || errors.Is(err, errInapplicable) != (name == "only excluded") {
			t.Errorf("%s receipt judged image evidence: %v", name, err)
		}
	}
	if err := mediaCurrent(testReport{Passed: false}, "video"); err == nil || !strings.Contains(err.Error(), "failed") {
		t.Errorf("a failed run's refusal does not say so: %v", err)
	}
}
