package evaluation

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/textcheck"
)

const (
	// ActivationProfileMediaType identifies typed activation profiles.
	ActivationProfileMediaType = "application/vnd.overgo.activation-profile+json"
	// ActivationProfileSchema identifies the activation profile contract.
	ActivationProfileSchema = "overgo/activation-profile/v1"
	// ActivationMatrixMediaType identifies activation coverage evidence.
	ActivationMatrixMediaType = "application/vnd.overgo.activation-matrix+json"
	// ActivationMatrixSchema identifies the activation matrix contract.
	ActivationMatrixSchema = "overgo/activation-matrix/v1"
	// ActivationMatrixAlias names the current coverage denominator.
	ActivationMatrixAlias = "evaluation/activation-matrix/active"
)

// ActivationProfileKind names a governed execution surface.
type ActivationProfileKind string

const (
	// ActivationProfileLocalModel is local model execution.
	ActivationProfileLocalModel ActivationProfileKind = "local-model"
	// ActivationProfilePeerModel is peer model execution.
	ActivationProfilePeerModel ActivationProfileKind = "peer-model"
	// ActivationProfileAgentTool is agent-tool execution.
	ActivationProfileAgentTool ActivationProfileKind = "agent-tool"
)

// ActivationProfile binds applicable tasks to one exact capability.
type ActivationProfile struct {
	Version    uint16                `json:"version"`
	Name       string                `json:"name"`
	Kind       ActivationProfileKind `json:"kind"`
	Capability artifact.ID           `json:"capability"`
	Tasks      []recipe.Task         `json:"tasks"`
	ID         artifact.ID           `json:"-"`
}

// ActivationCoverageResult supplies evidence or an explicit gap for one cell.
type ActivationCoverageResult struct {
	Case     artifact.ID `json:"case"`
	Profile  artifact.ID `json:"profile"`
	Evidence artifact.ID `json:"evidence,omitzero"`
	Gap      string      `json:"gap,omitzero"`
}

// ActivationMatrixCell is one case-profile coverage fact.
type ActivationMatrixCell struct {
	Case     artifact.ID `json:"case"`
	Profile  artifact.ID `json:"profile"`
	Evidence artifact.ID `json:"evidence,omitzero"`
	Gap      string      `json:"gap,omitzero"`
}

// ActivationMatrix records the full applicable cross-product and honest totals.
type ActivationMatrix struct {
	Version     uint16                 `json:"version"`
	Registry    artifact.ID            `json:"registry"`
	Profiles    []artifact.ID          `json:"profiles"`
	Cells       []ActivationMatrixCell `json:"cells"`
	Denominator uint32                 `json:"denominator"`
	Covered     uint32                 `json:"covered"`
	ID          artifact.ID            `json:"-"`
}

var activationProfileCodec = artifact.JSONDocumentCodec(
	"activation profile", artifact.KindProfile, ActivationProfileMediaType, ActivationProfileSchema,
	canonicalizeActivationProfile,
	func(value ActivationProfile) artifact.ID { return value.ID },
	func(value *ActivationProfile, id artifact.ID) { value.ID = id },
	func(value ActivationProfile) ActivationProfile { value.Tasks = slices.Clone(value.Tasks); return value },
)

var activationMatrixCodec = artifact.JSONDocumentCodec(
	"activation matrix", artifact.KindEvidence, ActivationMatrixMediaType, ActivationMatrixSchema,
	canonicalizeActivationMatrix,
	func(value ActivationMatrix) artifact.ID { return value.ID },
	func(value *ActivationMatrix, id artifact.ID) { value.ID = id },
	func(value ActivationMatrix) ActivationMatrix {
		value.Profiles = slices.Clone(value.Profiles)
		value.Cells = slices.Clone(value.Cells)
		return value
	},
)

// NewActivationProfile validates and identifies one exact activation surface.
func NewActivationProfile(value ActivationProfile) (ActivationProfile, error) {
	var unset uint16
	switch value.Version {
	case unset:
		value.Version = artifact.InitialDocumentVersion
	case artifact.InitialDocumentVersion:
	default:
		return ActivationProfile{}, errors.New("evaluation: unsupported activation profile version")
	}
	value.ID = artifact.ID{}
	identified, err := activationProfileCodec.New(value)
	if err != nil {
		return ActivationProfile{}, err
	}
	return identified, nil
}

// PublishActivationMatrix publishes profiles and their complete applicable matrix.
func PublishActivationMatrix(
	ctx context.Context,
	repository artifact.Repository,
	profiles []ActivationProfile,
	results []ActivationCoverageResult,
) (ActivationMatrix, artifact.CommitID, error) {
	registry, cases, found, err := ResolveActivationCaseRegistry(ctx, repository)
	if err != nil || !found || len(profiles) == 0 {
		return ActivationMatrix{}, artifact.CommitID{}, errors.Join(errors.New("evaluation: activation matrix authority is absent"), err)
	}
	identified := make([]ActivationProfile, len(profiles))
	for index, profile := range profiles {
		identified[index], err = NewActivationProfile(profile)
		if err != nil {
			return ActivationMatrix{}, artifact.CommitID{}, err
		}
		if _, err := runrecord.RequireCapabilityIdentity(ctx, repository, identified[index].Capability); err != nil {
			return ActivationMatrix{}, artifact.CommitID{}, err
		}
	}
	slices.SortFunc(identified, func(left, right ActivationProfile) int { return strings.Compare(left.Name, right.Name) })
	for index := range identified {
		if index > 0 && identified[index-1].Name == identified[index].Name {
			return ActivationMatrix{}, artifact.CommitID{}, errors.New("evaluation: duplicate activation profile")
		}
	}
	resultByCell := make(map[[2]artifact.ID]ActivationCoverageResult, len(results))
	for _, result := range results {
		key := [2]artifact.ID{result.Case, result.Profile}
		if _, duplicate := resultByCell[key]; duplicate {
			return ActivationMatrix{}, artifact.CommitID{}, errors.New("evaluation: duplicate activation coverage result")
		}
		resultByCell[key] = result
	}
	contents := make([]artifact.Content, 0, len(identified)+1)
	lineage := []artifact.Lineage{}
	profileIDs := make([]artifact.ID, len(identified))
	var cells []ActivationMatrixCell
	var covered uint32
	for index, profile := range identified {
		content, err := activationProfileCodec.Content(profile)
		if err != nil {
			return ActivationMatrix{}, artifact.CommitID{}, err
		}
		contents = append(contents, content)
		profileIDs[index] = profile.ID
		lineage = append(lineage, artifact.DependencyLineage(profile.ID, profile.Capability)...)
		profileCells := len(cells)
		for _, testCase := range cases {
			if !slices.Contains(profile.Tasks, testCase.Task) {
				continue
			}
			result, supplied := resultByCell[[2]artifact.ID{testCase.ID, profile.ID}]
			cell := ActivationMatrixCell{Case: testCase.ID, Profile: profile.ID, Gap: "missing-evidence"}
			if supplied {
				cell.Evidence, cell.Gap = result.Evidence, result.Gap
				delete(resultByCell, [2]artifact.ID{testCase.ID, profile.ID})
			}
			if cell.Evidence.Valid() {
				covered++
			}
			cells = append(cells, cell)
		}
		if len(cells) == profileCells {
			return ActivationMatrix{}, artifact.CommitID{}, errors.New("evaluation: activation profile applies to no registered cases")
		}
	}
	if len(resultByCell) != 0 {
		return ActivationMatrix{}, artifact.CommitID{}, errors.New("evaluation: coverage result is outside the activation matrix")
	}
	matrix, err := activationMatrixCodec.New(ActivationMatrix{
		Version: artifact.InitialDocumentVersion, Registry: registry.ID, Profiles: profileIDs,
		Cells: cells, Denominator: uint32(len(cells)), Covered: covered,
	})
	if err != nil {
		return ActivationMatrix{}, artifact.CommitID{}, err
	}
	matrixContent, err := activationMatrixCodec.Content(matrix)
	if err != nil {
		return ActivationMatrix{}, artifact.CommitID{}, err
	}
	contents = append(contents, matrixContent)
	parents := append([]artifact.ID{registry.ID}, profileIDs...)
	for _, cell := range matrix.Cells {
		parents = append(parents, cell.Case)
		if cell.Evidence.Valid() {
			parents = append(parents, cell.Evidence)
		}
	}
	lineage = append(lineage, artifact.DependencyLineage(matrix.ID, parents...)...)
	previous, active, err := artifact.ResolveAlias(ctx, repository, ActivationMatrixAlias)
	if err != nil {
		return ActivationMatrix{}, artifact.CommitID{}, err
	}
	if active && previous == matrix.ID {
		commit, _ := repository.Head()
		return matrix, commit, nil
	}
	binding := artifact.AliasBinding{Name: ActivationMatrixAlias, Target: matrix.ID}
	if active {
		binding.Previous = artifact.CloneID(&previous)
	}
	batch, err := artifact.NewDocumentBatch("evaluation/activation-matrix/"+matrix.ID.String(), contents, lineage, []artifact.AliasBinding{binding})
	if err != nil {
		return ActivationMatrix{}, artifact.CommitID{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	return matrix, commit, err
}

// ResolveActivationMatrix loads the current matrix.
func ResolveActivationMatrix(ctx context.Context, reader artifact.Reader) (ActivationMatrix, bool, error) {
	return activationMatrixCodec.Resolve(ctx, reader, ActivationMatrixAlias)
}

// AdmitActivationProfile requires enumeration and complete evidence coverage.
func AdmitActivationProfile(matrix ActivationMatrix, profile artifact.ID) error {
	if err := activationMatrixCodec.ValidateIdentity(matrix); err != nil || !slices.Contains(matrix.Profiles, profile) {
		return errors.Join(errors.New("evaluation: activation profile is not enumerated"), err)
	}
	for _, cell := range matrix.Cells {
		if cell.Profile == profile && !cell.Evidence.Valid() {
			return errors.New("evaluation: activation profile has explicit coverage gaps")
		}
	}
	return nil
}

func canonicalizeActivationProfile(value *ActivationProfile) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Name == "" || strings.TrimSpace(value.Name) != value.Name ||
		!textcheck.Bounded(value.Name, len(value.Name), "\x00\r\n\t") || value.Capability.Kind() != artifact.KindProfile || len(value.Tasks) == 0 ||
		value.Kind != ActivationProfileLocalModel && value.Kind != ActivationProfilePeerModel && value.Kind != ActivationProfileAgentTool {
		return errors.New("evaluation: invalid activation profile")
	}
	slices.Sort(value.Tasks)
	if slices.ContainsFunc(value.Tasks, func(task recipe.Task) bool { return !task.Valid() }) || len(slices.Compact(slices.Clone(value.Tasks))) != len(value.Tasks) {
		return errors.New("evaluation: invalid activation profile tasks")
	}
	return nil
}

func canonicalizeActivationMatrix(value *ActivationMatrix) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Registry.Kind() != artifact.KindProfile ||
		len(value.Profiles) == 0 || len(value.Cells) == 0 || uint32(len(value.Cells)) != value.Denominator || value.Covered > value.Denominator {
		return errors.New("evaluation: invalid activation matrix")
	}
	slices.SortFunc(value.Profiles, artifact.CompareID)
	slices.SortFunc(value.Cells, func(left, right ActivationMatrixCell) int {
		if order := artifact.CompareID(left.Profile, right.Profile); order != 0 {
			return order
		}
		return artifact.CompareID(left.Case, right.Case)
	})
	seen := map[[2]artifact.ID]bool{}
	covered := uint32(0)
	for _, cell := range value.Cells {
		key := [2]artifact.ID{cell.Case, cell.Profile}
		if cell.Case.Kind() != artifact.KindRecipe || !slices.Contains(value.Profiles, cell.Profile) || seen[key] ||
			cell.Evidence.Valid() == (cell.Gap != "") || cell.Evidence.Valid() && cell.Evidence.Kind() != artifact.KindEvidence ||
			cell.Gap != "" && (!textcheck.Bounded(cell.Gap, len(cell.Gap), "\x00\r\n\t") || strings.TrimSpace(cell.Gap) != cell.Gap) {
			return errors.New("evaluation: invalid activation matrix cell")
		}
		seen[key] = true
		if cell.Evidence.Valid() {
			covered++
		}
	}
	if covered != value.Covered || len(slices.Compact(slices.Clone(value.Profiles))) != len(value.Profiles) {
		return errors.New("evaluation: activation matrix totals differ")
	}
	return nil
}
