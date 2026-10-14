package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/textcheck"
)

const (
	// activationCaseMediaType identifies behavioral activation cases.
	activationCaseMediaType = "application/vnd.overgo.activation-case+json"
	// activationCaseSchema identifies the behavioral case contract.
	activationCaseSchema = "overgo/activation-case/v1"
	// activationCaseRegistryMediaType identifies case denominator snapshots.
	activationCaseRegistryMediaType = "application/vnd.overgo.activation-case-registry+json"
	// activationCaseRegistrySchema identifies the case registry contract.
	activationCaseRegistrySchema = "overgo/activation-case-registry/v1"
	// activationCaseRegistryAlias names the current registered denominator.
	activationCaseRegistryAlias = "evaluation/activation-cases/active"
)

// ActivationCase declares one task contract and its evidence check without
// naming any local, peer, model, or tool activation profile.
type ActivationCase struct {
	Version       uint16      `json:"version"`
	Name          string      `json:"name"`
	Task          recipe.Task `json:"task"`
	Contract      artifact.ID `json:"contract"`
	Input         artifact.ID `json:"input"`
	EvidenceCheck artifact.ID `json:"evidence_check"`
	ID            artifact.ID `json:"-"`
}

// activationCaseRegistry is the exact, store-published case denominator.
type activationCaseRegistry struct {
	Version     uint16        `json:"version"`
	Cases       []artifact.ID `json:"cases"`
	Denominator uint32        `json:"denominator"`
	ID          artifact.ID   `json:"-"`
}

var activationCaseCodec = artifact.JSONDocumentCodec(
	"activation case", artifact.KindRecipe, activationCaseMediaType, activationCaseSchema,
	canonicalizeActivationCase,
	func(value ActivationCase) artifact.ID { return value.ID },
	func(value *ActivationCase, id artifact.ID) { value.ID = id },
)

var activationCaseRegistryCodec = artifact.JSONDocumentCodec(
	"activation case registry", artifact.KindProfile, activationCaseRegistryMediaType, activationCaseRegistrySchema,
	canonicalizeActivationCaseRegistry,
	func(value activationCaseRegistry) artifact.ID { return value.ID },
	func(value *activationCaseRegistry, id artifact.ID) { value.ID = id },
)

// NewActivationCase validates and identifies one profile-independent case.
func NewActivationCase(value ActivationCase) (ActivationCase, error) {
	if value.Version != 0 && value.Version != artifact.InitialDocumentVersion {
		return ActivationCase{}, errors.New("evaluation: unsupported activation case version")
	}
	return activationCaseCodec.NewInitial(value)
}

// PublishActivationCaseRegistry atomically publishes cases and advances the
// current denominator alias.
func PublishActivationCaseRegistry(
	ctx context.Context,
	repository artifact.Repository,
	cases []ActivationCase,
) (activationCaseRegistry, artifact.CommitID, error) {
	if ctx == nil || repository == nil || len(cases) == 0 || uint64(len(cases)) > uint64(math.MaxUint32) {
		return activationCaseRegistry{}, artifact.CommitID{}, errors.New("evaluation: activation case registry authority is absent or bounded")
	}
	identified := make([]ActivationCase, len(cases))
	for index, value := range cases {
		var err error
		identified[index], err = NewActivationCase(value)
		if err != nil {
			return activationCaseRegistry{}, artifact.CommitID{}, err
		}
	}
	slices.SortFunc(identified, func(left, right ActivationCase) int { return strings.Compare(left.Name, right.Name) })
	contents := make([]artifact.Content, 0, len(identified)+1)
	lineage := []artifact.Lineage{}
	caseIDs := make([]artifact.ID, len(identified))
	for index, value := range identified {
		if index > 0 && identified[index-1].Name == value.Name {
			return activationCaseRegistry{}, artifact.CommitID{}, errors.New("evaluation: duplicate activation case name")
		}
		content, err := activationCaseCodec.Content(value)
		if err != nil {
			return activationCaseRegistry{}, artifact.CommitID{}, err
		}
		contents = append(contents, content)
		caseIDs[index] = value.ID
		lineage = append(lineage, artifact.DependencyLineage(value.ID, value.Contract, value.Input, value.EvidenceCheck)...)
	}
	registry, err := activationCaseRegistryCodec.New(activationCaseRegistry{
		Version: artifact.InitialDocumentVersion, Cases: caseIDs, Denominator: uint32(len(caseIDs)),
	})
	if err != nil {
		return activationCaseRegistry{}, artifact.CommitID{}, err
	}
	content, err := activationCaseRegistryCodec.Content(registry)
	if err != nil {
		return activationCaseRegistry{}, artifact.CommitID{}, err
	}
	contents = append(contents, content)
	lineage = append(lineage, artifact.DependencyLineage(registry.ID, caseIDs...)...)
	previous, found, err := artifact.ResolveAlias(ctx, repository, activationCaseRegistryAlias)
	if err != nil {
		return activationCaseRegistry{}, artifact.CommitID{}, err
	}
	if found && previous == registry.ID {
		commit, _ := repository.Head()
		return registry, commit, nil
	}
	binding := artifact.AliasBinding{Name: activationCaseRegistryAlias, Target: registry.ID}
	if found {
		binding = artifact.AliasMove(binding.Name, binding.Target, previous)
	}
	batch, err := artifact.NewDocumentBatch("evaluation/activation-cases/"+registry.ID.String(), contents, lineage, []artifact.AliasBinding{binding})
	if err != nil {
		return activationCaseRegistry{}, artifact.CommitID{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	return registry, commit, err
}

// ResolveActivationCaseRegistry loads the current denominator and every exact
// case it contains.
func ResolveActivationCaseRegistry(
	ctx context.Context,
	reader artifact.Reader,
) (activationCaseRegistry, []ActivationCase, bool, error) {
	registry, found, err := activationCaseRegistryCodec.Resolve(ctx, reader, activationCaseRegistryAlias)
	if err != nil || !found {
		return activationCaseRegistry{}, nil, found, err
	}
	cases := make([]ActivationCase, len(registry.Cases))
	for index, id := range registry.Cases {
		cases[index], err = activationCaseCodec.Require(ctx, reader, id)
		if err != nil {
			return activationCaseRegistry{}, nil, false, err
		}
	}
	return registry, cases, true, nil
}

func canonicalizeActivationCase(value *ActivationCase) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || strings.TrimSpace(value.Name) != value.Name ||
		value.Name == "" || !textcheck.Bounded(value.Name, len(value.Name), "\x00\r\n\t") || !value.Task.Valid() ||
		value.Contract.Kind() != artifact.KindRecipe || value.Input.Kind() != artifact.KindDataset ||
		value.EvidenceCheck.Kind() != artifact.KindProfile {
		return errors.New("evaluation: invalid activation case")
	}
	return nil
}

func canonicalizeActivationCaseRegistry(value *activationCaseRegistry) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || len(value.Cases) == 0 ||
		uint64(len(value.Cases)) > uint64(math.MaxUint32) || uint32(len(value.Cases)) != value.Denominator {
		return errors.New("evaluation: invalid activation case registry")
	}
	if !slices.IsSortedFunc(value.Cases, artifact.CompareID) {
		slices.SortFunc(value.Cases, artifact.CompareID)
	}
	if slices.ContainsFunc(value.Cases, func(id artifact.ID) bool { return id.Kind() != artifact.KindRecipe }) ||
		len(slices.Compact(slices.Clone(value.Cases))) != len(value.Cases) {
		return errors.New("evaluation: invalid activation case population")
	}
	return nil
}
