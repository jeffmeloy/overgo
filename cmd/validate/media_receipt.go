package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/processcontrol"
)

// Media and specialized evidence is keyed to the media runtime identity, and
// whether it holds at a tree is decided by the acceptance tests of one package:
// the chain of reviewed layers they walk is the only owner of that judgement.
// The planner runs those acceptances with go test at this tree and reads the
// named verdicts; go test's cache, keyed to the exact inputs, replays a run
// whose inputs did not change, so asking again costs nothing.

// mediaAcceptancePackage is the package whose complete receipt judges media
// and specialized evidence.
const mediaAcceptancePackage = "cmd/compatibility"

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
// whether it passed, and each test's final action.
type testReport struct {
	Passed bool
	Tests  map[string]string
}

// packageTests runs the acceptances the judged surfaces name in one package
// at the working tree.
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
		Path: "go", Args: []string{"test", "-json", "-run", strings.Join(slices.Compact(patterns), "|"), "./" + packagePath}, Dir: root,
		Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return testReport{}, fmt.Errorf("validate: test %s: %w", packagePath, err)
	}
	report := testReport{Passed: receipt.ExitCode == 0, Tests: map[string]string{}}
	decoder := json.NewDecoder(&stdout)
	for {
		var event struct{ Action, Test string }
		if err := decoder.Decode(&event); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return testReport{}, fmt.Errorf("validate: test %s: %w: %s", packagePath, err, strings.TrimSpace(stderr.String()))
		}
		if event.Test != "" && (event.Action == "pass" || event.Action == "fail" || event.Action == "skip") {
			report.Tests[event.Test] = event.Action
		}
	}
	return report, nil
}

// mediaCurrent says why a surface's evidence is not current at this tree: the
// acceptance run failed, or the surface's acceptances are not among the
// verdicts that passed. Nil means the
// named acceptances passed against exactly this tree's inputs.
func mediaCurrent(report testReport, surface SurfaceID) error {
	pattern, judged := surfaceAcceptances[surface]
	pkg := surfacePackages[surface]
	if !judged || pkg == "" {
		return errCellNotJudged
	}
	rerun := "run: go test ./" + pkg
	if !report.Passed {
		return fmt.Errorf("the complete run of %s failed at this tree's inputs; repair it, then %s", pkg, rerun)
	}
	// A receipt carries a skip only for a declared exclusion, a test that
	// stated why it cannot apply at this tree; it neither judges the surface
	// nor refuses it. What judges is that at least one named acceptance
	// passed and none recorded anything else.
	matched := 0
	for name, action := range report.Tests {
		if !pattern.MatchString(name) || action == "skip" {
			continue
		}
		if action != "pass" {
			return fmt.Errorf("%s acceptance %s recorded %q, not a pass, at this tree's inputs", surface, name, action)
		}
		matched++
	}
	if matched == 0 {
		return fmt.Errorf("the receipt for %s names no passing %s acceptance among its verdicts", pkg, surface)
	}
	return nil
}
