package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"overgo/internal/processcontrol"
)

// Media and specialized evidence is keyed to the media runtime identity, and
// whether it holds at a tree is decided by the acceptance tests of one package:
// the chain of reviewed layers they walk is the only owner of that judgement.
// The gate keeps a receipt of that package's complete run, keyed to the exact
// inputs it ran against, so the planner asks the gate for the receipt at this
// tree and reads the named verdicts, instead of re-running the suite or
// duplicating the chain.

// mediaAcceptancePackage is the package whose complete receipt judges media
// and specialized evidence.
const mediaAcceptancePackage = "cmd/compatibility"

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
}

// gateReceipt is what cmd/gate -package-receipt reports.
type gateReceipt struct {
	Package    string            `json:"package"`
	Mode       string            `json:"mode"`
	Obligation string            `json:"obligation"`
	Found      bool              `json:"found"`
	Passed     bool              `json:"passed"`
	Tests      map[string]string `json:"tests"`
}

// gatePackageReceipt asks the gate for a package's complete receipt at the
// working tree.
func gatePackageReceipt(ctx context.Context, root, packagePath string) (gateReceipt, error) {
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"run", "./cmd/gate", "-package-receipt", packagePath}, Dir: root,
		Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return gateReceipt{}, fmt.Errorf("validate: gate package receipt: %w", err)
	}
	if receipt.ExitCode != 0 {
		return gateReceipt{}, fmt.Errorf("validate: gate package receipt: exit=%d: %s", receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	var report gateReceipt
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		return gateReceipt{}, fmt.Errorf("validate: gate package receipt: %w", err)
	}
	return report, nil
}

// mediaCurrent says why a surface's evidence is not current at this tree: the
// acceptance package has no receipt for these inputs, its run failed, or the
// surface's acceptances are not among the verdicts that passed. Nil means the
// named acceptances passed against exactly this tree's inputs.
func mediaCurrent(report gateReceipt, surface SurfaceID) error {
	pattern, judged := surfaceAcceptances[surface]
	if !judged {
		return errCellNotJudged
	}
	rerun := "the gate writes it when a landing selects " + mediaAcceptancePackage + " in complete mode"
	if !report.Found {
		return fmt.Errorf("no complete receipt for %s at this tree's inputs; %s", mediaAcceptancePackage, rerun)
	}
	if !report.Passed {
		return fmt.Errorf("the complete run of %s failed at this tree's inputs; repair it, then %s", mediaAcceptancePackage, rerun)
	}
	matched := 0
	for name, action := range report.Tests {
		if !pattern.MatchString(name) {
			continue
		}
		if action != "pass" {
			return fmt.Errorf("%s acceptance %s recorded %q, not a pass, at this tree's inputs", surface, name, action)
		}
		matched++
	}
	if matched == 0 {
		return fmt.Errorf("the receipt for %s names no %s acceptance among its verdicts", mediaAcceptancePackage, surface)
	}
	return nil
}
