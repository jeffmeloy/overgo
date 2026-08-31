package runrecord

import (
	"cmp"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"path"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
)

const (
	// ExternalMechanismProvenanceMediaType identifies external research provenance.
	ExternalMechanismProvenanceMediaType = "application/vnd.overgo.external-mechanism-provenance+json"
	// ExternalMechanismProvenanceSchema identifies the external provenance contract version.
	ExternalMechanismProvenanceSchema = "overgo/external-mechanism-provenance/v1"
)

const externalMechanismTransfer = "reexpress-overgo-native"

// ExternalMechanismSource binds one mechanism to exact paths within a source commit.
type ExternalMechanismSource struct {
	Mechanism artifact.ID `json:"mechanism"`
	Paths     []string    `json:"paths"`
}

// ExternalMechanismProvenance binds a census to immutable external source and license authority.
type ExternalMechanismProvenance struct {
	Version    uint16                    `json:"version"`
	Census     artifact.ID               `json:"census"`
	Repository string                    `json:"repository"`
	Commit     string                    `json:"commit"`
	License    artifact.ID               `json:"license"`
	Transfer   string                    `json:"transfer"`
	Sources    []ExternalMechanismSource `json:"sources"`
	ID         artifact.ID               `json:"-"`
}

var externalMechanismProvenanceCodec = artifact.JSONDocumentCodec(
	"external mechanism provenance", artifact.KindEvidence,
	ExternalMechanismProvenanceMediaType, ExternalMechanismProvenanceSchema,
	canonicalizeExternalMechanismProvenance,
	func(value ExternalMechanismProvenance) artifact.ID { return value.ID },
	func(value *ExternalMechanismProvenance, id artifact.ID) { value.ID = id },
	func(value ExternalMechanismProvenance) ExternalMechanismProvenance {
		value.Sources = slices.Clone(value.Sources)
		for index := range value.Sources {
			value.Sources[index].Paths = slices.Clone(value.Sources[index].Paths)
		}
		return value
	},
)

// NewExternalMechanismProvenance identifies exact research-only source provenance.
func NewExternalMechanismProvenance(
	census trainingprogram.MechanismCensus,
	repository, commit string,
	license artifact.ID,
	sources []ExternalMechanismSource,
) (ExternalMechanismProvenance, error) {
	mechanisms := make(map[artifact.ID]struct{}, len(census.Assessments()))
	for _, assessment := range census.Assessments() {
		mechanisms[assessment.Mechanism] = struct{}{}
	}
	for _, source := range sources {
		if _, found := mechanisms[source.Mechanism]; !found {
			return ExternalMechanismProvenance{}, errors.New("run record: external source names mechanism outside census")
		}
		delete(mechanisms, source.Mechanism)
	}
	if len(mechanisms) != 0 {
		return ExternalMechanismProvenance{}, errors.New("run record: external source coverage differs from census")
	}
	return externalMechanismProvenanceCodec.New(ExternalMechanismProvenance{
		Version: artifact.InitialDocumentVersion, Census: census.ID(), Repository: repository,
		Commit: commit, License: license, Transfer: externalMechanismTransfer, Sources: sources,
	})
}

// RequireExternalMechanismProvenance loads exact provenance through its owning contract.
func RequireExternalMechanismProvenance(
	ctx context.Context, reader artifact.Reader, id artifact.ID,
) (ExternalMechanismProvenance, error) {
	return externalMechanismProvenanceCodec.Require(ctx, reader, id)
}

// Content returns the canonical external provenance document.
func (value ExternalMechanismProvenance) Content() (artifact.Content, error) {
	return externalMechanismProvenanceCodec.Content(value)
}

// Lineage binds provenance to the census, license, and each mechanism.
func (value ExternalMechanismProvenance) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Census, value.License}
	for _, source := range value.Sources {
		parents = append(parents, source.Mechanism)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func canonicalizeExternalMechanismProvenance(value *ExternalMechanismProvenance) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Census.Kind() != artifact.KindRecipe ||
		value.License.Kind() != artifact.KindEvidence || value.Transfer != externalMechanismTransfer ||
		!validExternalSourceText(value.Repository) || !validExternalCommit(value.Commit) || len(value.Sources) == 0 {
		return errors.New("run record: invalid external mechanism provenance")
	}
	value.Sources = slices.Clone(value.Sources)
	slices.SortFunc(value.Sources, func(left, right ExternalMechanismSource) int {
		return cmp.Compare(left.Mechanism.String(), right.Mechanism.String())
	})
	for index := range value.Sources {
		source := &value.Sources[index]
		if source.Mechanism.Kind() != artifact.KindRecipe || len(source.Paths) == 0 ||
			index > 0 && value.Sources[index-1].Mechanism == source.Mechanism {
			return errors.New("run record: invalid external mechanism source")
		}
		source.Paths = slices.Clone(source.Paths)
		slices.Sort(source.Paths)
		for pathIndex, sourcePath := range source.Paths {
			if !validExternalPath(sourcePath) || pathIndex > 0 && source.Paths[pathIndex-1] == sourcePath {
				return errors.New("run record: invalid external mechanism path")
			}
		}
	}
	return nil
}

func validExternalCommit(value string) bool {
	if value == "" || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && (len(decoded) == sha1.Size || len(decoded) == sha256.Size)
}

func validExternalSourceText(value string) bool {
	return validText(value)
}

func validExternalPath(value string) bool {
	return validExternalSourceText(value) && !strings.Contains(value, "\\") && path.Clean(value) == value && !path.IsAbs(value) && value != "." && value != ".." && !strings.HasPrefix(value, "../")
}
