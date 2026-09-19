package gate

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

const gatePreparedSelectionFile = "tmp/gate_prepared_selection.json"

// preparedSelection is the compact, source-bound plan a deferring gate leaves
// for its lane runner: the immutable manifest it already composed, the exact
// candidate tree that manifest was composed against, and the preparation that
// owns it. It carries no AST or type graph -- the manifest already holds the
// resolved selection and its planned invocations.
type preparedSelection struct {
	Version       uint16                       `json:"version"`
	Preparation   artifact.ID                  `json:"preparation"`
	CandidateTree string                       `json:"candidate_tree"`
	ManifestID    artifact.ID                  `json:"manifest_id"`
	Manifest      automationcheck.ManifestPlan `json:"manifest"`
}

// persistPreparedSelection records the composed selection beside the lane
// locator, so the runner can resume from it instead of recomposing the manifest.
func (g *gateContext) persistPreparedSelection() error {
	if g.manifestPlan == nil || g.acceptedTree == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Join(g.repo, filepath.Dir(filepath.FromSlash(gatePreparedSelectionFile))), clioptions.OutputDirectoryMode); err != nil {
		return err
	}
	return writeJSON(g.repo, gatePreparedSelectionFile, preparedSelection{
		Version: artifact.InitialDocumentVersion, Preparation: g.preparation.ID,
		CandidateTree: g.acceptedTree, ManifestID: g.manifestPlan.ID, Manifest: *g.manifestPlan,
	}, clioptions.OutputFileMode)
}

// readPreparedSelection loads the selection a deferring gate left; an absent or
// corrupt file yields no selection, so the runner reconstructs conservatively.
func readPreparedSelection(repo string) (preparedSelection, bool, error) {
	data, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(gatePreparedSelectionFile)))
	if errors.Is(err, os.ErrNotExist) {
		return preparedSelection{}, false, nil
	}
	if err != nil {
		return preparedSelection{}, false, err
	}
	var selection preparedSelection
	if err := json.Unmarshal(data, &selection); err != nil {
		return preparedSelection{}, false, nil
	}
	return selection, true, nil
}

// reusePreparedPipeline resumes the deferred lanes from the selection the
// deferring gate left, and reports whether it did. It resumes only when the
// selection belongs to this obligation, its manifest verifies, the landed tree
// differs from the composed candidate only by the advanced plan, and the cheaply
// rebuilt invocations are exactly the ones the manifest planned. Any missing or
// corrupt data, source difference or definition drift returns false so the
// caller reconstructs the pipeline.
func (g *gateContext) reusePreparedPipeline(obligation runrecord.GateLaneObligation, root string) (plannedPipeline, bool, error) {
	prepared, ok, err := readPreparedSelection(g.repo)
	if err != nil {
		return plannedPipeline{}, false, err
	}
	if !ok || prepared.Preparation != obligation.Preparation {
		return plannedPipeline{}, false, nil
	}
	manifest := prepared.Manifest
	manifest.ID = prepared.ManifestID
	if manifest.Validate() != nil {
		return plannedPipeline{}, false, nil
	}
	if candidateTreeKey(g.fixedTree) != manifest.CandidateTree && !g.landedTreeMatchesCandidate(prepared.CandidateTree) {
		return plannedPipeline{}, false, nil
	}
	definitions, err := g.preparedDefinitions(root)
	if err != nil {
		return plannedPipeline{}, false, nil
	}
	impact := automationcheck.Impact{Facts: slices.Clone(manifest.Facts), Exclusions: slices.Clone(manifest.Exclusions)}
	invocations, err := automationcheck.Plan(definitions, impact)
	if err != nil {
		return plannedPipeline{}, false, nil
	}
	planned := make(map[artifact.ID]bool, len(manifest.Invocations))
	for _, invocation := range manifest.Invocations {
		planned[invocation.ID] = true
	}
	for _, invocation := range invocations {
		if !planned[invocation.ID] {
			return plannedPipeline{}, false, nil
		}
	}
	g.manifestPlan = &manifest
	g.note("deferred lanes resumed from the prepared selection")
	return plannedPipeline{definitions: definitions, invocations: invocations, manifest: &manifest}, true, nil
}

// landedTreeMatchesCandidate reports whether the landed tree differs from the
// composed candidate only by the advanced plan document: the source the manifest
// selected against is then unchanged, so the selection still holds. Any other
// difference, or an unreadable diff, is treated as a mismatch.
func (g *gateContext) landedTreeMatchesCandidate(candidateTree string) bool {
	diff, err := gitLines(g.repo, "diff", "--name-only", candidateTree, g.fixedTree)
	if err != nil {
		return false
	}
	for _, path := range diff {
		if trimmed := strings.TrimSpace(path); trimmed != "" && trimmed != plan.Path {
			return false
		}
	}
	return true
}

// preparedDefinitions rebuilds the check definitions from their cheap owners,
// with no manifest re-derivation: the device lane's packages, the verification
// batch and the deferred-lane rewiring the composed manifest already reflects.
func (g *gateContext) preparedDefinitions(root string) ([]automationcheck.Check, error) {
	verificationBatch, err := planVerificationBatch(root, g.planRef)
	if err != nil {
		return nil, err
	}
	devicePackages := automationcheck.DeviceLanePackages()
	if devicePlan, err := automationcheck.DevicePlan(root, g.paths); err == nil {
		for _, packagePath := range devicePlan.Packages {
			devicePackages = append(devicePackages, strings.TrimPrefix(packagePath, "./"))
		}
	}
	slices.Sort(devicePackages)
	devicePackages = slices.Compact(devicePackages)
	definitions := g.pipelineChecks(devicePackages...)
	definitions, err = g.batchAcceptanceChecks(definitions, verificationBatch)
	if err != nil {
		return nil, err
	}
	return g.rewireDeferredLanes(definitions), nil
}
