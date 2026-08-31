package automationcheck

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/codemanifest"
)

const (
	// ManifestAnalysisMediaType identifies immutable selector-analysis evidence.
	ManifestAnalysisMediaType = "application/vnd.overgo.code-manifest-analysis+json"
	// ManifestAnalysisSchema identifies the strict manifest-analysis contract.
	ManifestAnalysisSchema = "overgo/code-manifest-analysis/v1"
)

// ManifestAnalysis binds structural change, impact, selection, and measured
// execution to the exact immutable verification plan that joined them.
type ManifestAnalysis struct {
	Version      uint16               `json:"version"`
	Delta        codemanifest.Delta   `json:"delta"`
	Impact       codemanifest.Impact  `json:"impact"`
	Plan         ManifestPlan         `json:"plan"`
	Selection    SelectionMetrics     `json:"selection"`
	Measurements ManifestMeasurements `json:"measurements"`
	ID           artifact.ID          `json:"-"`
}

var manifestAnalysisCodec = artifact.JSONDocumentCodec(
	"automation manifest analysis", artifact.KindEvidence, ManifestAnalysisMediaType, ManifestAnalysisSchema,
	validateManifestAnalysis, func(value ManifestAnalysis) artifact.ID { return value.ID },
	func(value *ManifestAnalysis, id artifact.ID) { value.ID = id }, cloneManifestAnalysis,
)

// NewManifestAnalysis validates and identifies one analysis document.
func NewManifestAnalysis(
	delta codemanifest.Delta,
	impact codemanifest.Impact,
	plan ManifestPlan,
	selection SelectionMetrics,
	measurements ManifestMeasurements,
) (ManifestAnalysis, error) {
	return manifestAnalysisCodec.NewInitial(ManifestAnalysis{
		Delta: delta, Impact: impact,
		Plan: plan, Selection: selection, Measurements: measurements,
	})
}

// ParseManifestAnalysis strictly decodes canonical analysis evidence.
func ParseManifestAnalysis(data []byte) (ManifestAnalysis, error) {
	return manifestAnalysisCodec.Parse(data)
}

// ValidateIdentity verifies analysis structure and content identity.
func (analysis ManifestAnalysis) ValidateIdentity() error {
	return manifestAnalysisCodec.ValidateIdentity(analysis)
}

// Content returns canonical content for immutable publication.
func (analysis ManifestAnalysis) Content() (artifact.Content, error) {
	return manifestAnalysisCodec.Content(analysis)
}

func validateManifestAnalysis(analysis *ManifestAnalysis) error {
	if analysis == nil || analysis.Version != artifact.InitialDocumentVersion {
		return errors.New("automation manifest analysis: invalid version")
	}
	if !analysis.Plan.ID.Valid() {
		canonicalizeManifestPlan(&analysis.Plan)
		id, err := manifestPlanID(analysis.Plan)
		if err != nil {
			return errors.New("automation manifest analysis: invalid plan")
		}
		analysis.Plan.ID = id
	}
	if err := analysis.Plan.Validate(); err != nil {
		return errors.New("automation manifest analysis: invalid plan")
	}
	if analysis.Delta.Base != analysis.Plan.BaseManifest ||
		analysis.Delta.Candidate != analysis.Plan.CandidateManifest ||
		analysis.Impact.Base != analysis.Plan.BaseManifest.String() ||
		analysis.Impact.Candidate != analysis.Plan.CandidateManifest.String() {
		return errors.New("automation manifest analysis: structural authority mismatch")
	}
	selection := analysis.Selection
	if selection.Owned < 0 || selection.Triggered < 0 || selection.Excluded < 0 || selection.Unresolved < 0 ||
		selection.Triggered+selection.Excluded+selection.Unresolved != selection.Owned {
		return errors.New("automation manifest analysis: invalid selection accounting")
	}
	measurements := analysis.Measurements
	if measurements.Defined < 0 || measurements.Selected < 0 || measurements.Excluded < 0 ||
		measurements.Uncertainty < 0 || measurements.CacheEligible < 0 || measurements.CacheHits < 0 || measurements.CacheMisses < 0 ||
		measurements.CacheHits+measurements.CacheMisses != measurements.CacheEligible ||
		measurements.PlanningNS < 0 || measurements.AnalysisAgeNS < 0 ||
		measurements.Selected != len(analysis.Plan.Invocations) ||
		measurements.FullPlanParity != (measurements.Selected+measurements.Excluded == measurements.Defined) {
		return errors.New("automation manifest analysis: invalid measurements")
	}
	return nil
}

func cloneManifestAnalysis(analysis ManifestAnalysis) ManifestAnalysis {
	analysis.Delta.Files = slices.Clone(analysis.Delta.Files)
	analysis.Delta.Symbols = slices.Clone(analysis.Delta.Symbols)
	analysis.Delta.ExternalInputs = slices.Clone(analysis.Delta.ExternalInputs)
	analysis.Delta.Uncertainty = slices.Clone(analysis.Delta.Uncertainty)
	analysis.Impact.Seeds = slices.Clone(analysis.Impact.Seeds)
	analysis.Impact.Reachable = slices.Clone(analysis.Impact.Reachable)
	analysis.Impact.Packages = slices.Clone(analysis.Impact.Packages)
	analysis.Impact.ExternalInputs = slices.Clone(analysis.Impact.ExternalInputs)
	analysis.Impact.Uncertainty = slices.Clone(analysis.Impact.Uncertainty)
	analysis.Plan = cloneManifestPlan(analysis.Plan)
	return analysis
}
