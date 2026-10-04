package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/landing"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

// Media and specialized evidence is keyed to the media runtime identity, and
// whether it holds at a tree is decided by the acceptance tests of one package:
// the chain of reviewed layers they walk is the only owner of that judgement.
// The planner runs those acceptances with go test at this tree and reads
// testevidence's package-qualified report of the run; go test's cache, keyed
// to the exact inputs, replays a run whose inputs did not change, so asking
// again costs nothing.

// mediaAcceptancePackage is the package whose complete receipt judges media
// and specialized evidence.
const mediaAcceptancePackage = "cmd/validate"

// speechAcceptancePackage is the package whose complete receipt judges speech
// evidence: the speech synthesizer's goldens, which load the speech model's
// artifact and hold its tokenizer, backbone, flow, codec and end-to-end audio
// to their references.
const speechAcceptancePackage = "internal/speechsynth"

// surfacePackages names, per judged surface, the package whose complete
// receipt carries its acceptances.
var surfacePackages = map[SurfaceID]string{
	"image": mediaAcceptancePackage, "video": mediaAcceptancePackage, "vqa": mediaAcceptancePackage,
	"specialized": mediaAcceptancePackage, "speech": speechAcceptancePackage,
}

// errCellNotJudged marks a cell whose evidence currency this command has no
// owner to ask about yet.
var errCellNotJudged = errors.New("evidence currency is not judged here")

// surfaceAcceptances names, per surface, the acceptance tests whose passing
// at this tree means the surface's evidence is current. A surface absent here
// is not judged.
var surfaceAcceptances = map[SurfaceID]*regexp.Regexp{
	"image":       regexp.MustCompile(`^(TestImageVideo.*Acceptance|TestMedia.*Reconciliation|TestAcceptedImageVideoEvidence)$`),
	"video":       regexp.MustCompile(`^(TestImageVideo.*Acceptance|TestMedia.*Reconciliation|TestAcceptedImageVideoEvidence)$`),
	"vqa":         regexp.MustCompile(`^(TestAcceptedRxBrainVQA|TestAcceptedTextVisionEvidence)$`),
	"specialized": regexp.MustCompile(`^TestAcceptedSpecializedTaskEvidence$`),
	"speech":      regexp.MustCompile(`^(TestSpeakE2EProducesReferenceAudio|TestGenerationGolden|TestMimiDecodeStagesGolden|TestBackboneStreamGolden|TestFlowOneStepGolden|TestTokenizerGoldenCases)$`),
}

// testReport is one go test run of an acceptance package's acceptances:
// whether go test exited zero, and testevidence's report of its events.
type testReport struct {
	Passed bool
	Report testevidence.GoTestReport
}

// packageTests runs the acceptances the judged surfaces name in one package
// at the working tree, built with the cycle's tags: acceptances that read the
// live store carry them, so no landing runs them.
func packageTests(ctx context.Context, root, packagePath string) (testReport, error) {
	var patterns []string
	for surface, pkg := range surfacePackages {
		if pkg == packagePath {
			patterns = append(patterns, "("+surfaceAcceptances[surface].String()+")")
		}
	}
	slices.Sort(patterns)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"test", "-json", "-tags", landing.CycleTags, "-run", strings.Join(slices.Compact(patterns), "|"), "./" + packagePath}, Dir: root,
		Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return testReport{}, fmt.Errorf("validate: test %s: %w", packagePath, err)
	}
	report, err := testevidence.GoTestJSONReport(stdout.String())
	if err != nil {
		return testReport{}, fmt.Errorf("validate: test %s: %w: %s", packagePath, err, strings.TrimSpace(stderr.String()))
	}
	return testReport{Passed: receipt.ExitCode == 0, Report: report}, nil
}

// mediaCurrent says why a surface's evidence is not current at this tree: the
// acceptance run failed, or its package's run proves none of the surface's
// acceptances passed, as requiredPass judges a run. A declared exclusion, a
// test that stated why it cannot apply at this tree, neither judges the
// surface nor refuses it; a skip that declared nothing refuses it, and a
// same-named test of another package lends it nothing. Nil means a named
// acceptance passed against exactly this tree's inputs.
func mediaCurrent(receipt testReport, surface SurfaceID) error {
	pattern, judged := surfaceAcceptances[surface]
	pkg := surfacePackages[surface]
	if !judged || pkg == "" {
		return errCellNotJudged
	}
	rerun := "run: go test ./" + pkg
	if !receipt.Passed {
		return fmt.Errorf("the complete run of %s failed at this tree's inputs; repair it, then %s", pkg, rerun)
	}
	if err := requiredPass(receipt.Report, []string{"./" + pkg}, pattern.MatchString); err != nil {
		return fmt.Errorf("the receipt for %s proves no passing %s acceptance at this tree's inputs: %w", pkg, surface, err)
	}
	return nil
}
