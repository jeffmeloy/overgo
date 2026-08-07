package modelrecipe

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/strictjson"
)

const (
	CatalogProfileDerivationVersion   uint16 = 1
	CatalogProfileDerivationMediaType        = "application/vnd.overgo.profile-catalog-derivation+json"
	CatalogProfileDerivationSchema           = "overgo/profile-catalog-derivation/v1"
	catalogProfileSource                     = "embedded.architecture_catalog"
)

var catalogProfileDerivationContract = artifact.DocumentContract{
	Kind:      artifact.KindEvidence,
	MediaType: CatalogProfileDerivationMediaType,
	Schema:    CatalogProfileDerivationSchema,
}

type catalogProfileDerivationBody struct {
	Version          uint16 `json:"version"`
	ProfileVersion   uint16 `json:"profile_version"`
	Architecture     string `json:"architecture"`
	Source           string `json:"source"`
	FactSchemaSHA256 string `json:"fact_schema_sha256"`
	PolicySHA256     string `json:"policy_sha256"`
}

// CatalogProfileDerivation: immutable catalog-to-profile derivation fact.
type CatalogProfileDerivation struct {
	ID artifact.ID
	catalogProfileDerivationBody
}

func NewCatalogProfileDerivation(profile model.ArchitectureProfile) (CatalogProfileDerivation, error) {
	policy, err := json.Marshal(profile)
	if err != nil {
		return CatalogProfileDerivation{}, fmt.Errorf("model recipe: encode catalog profile policy: %w", err)
	}
	schema, err := json.Marshal(architectureProfileFactSpecs)
	if err != nil {
		return CatalogProfileDerivation{}, fmt.Errorf("model recipe: encode profile fact schema: %w", err)
	}
	document := CatalogProfileDerivation{catalogProfileDerivationBody: catalogProfileDerivationBody{
		Version: CatalogProfileDerivationVersion, ProfileVersion: ProfileVersion,
		Architecture: profile.Name, Source: catalogProfileSource,
		FactSchemaSHA256: sha256Hex(schema), PolicySHA256: sha256Hex(policy),
	}}
	if err := document.validateShape(); err != nil {
		return CatalogProfileDerivation{}, err
	}
	content, err := catalogProfileDerivationContent(document)
	if err != nil {
		return CatalogProfileDerivation{}, err
	}
	document.ID, err = catalogProfileDerivationContract.Identify(content)
	return document, err
}

func ParseCatalogProfileDerivation(content []byte) (CatalogProfileDerivation, error) {
	var body catalogProfileDerivationBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return CatalogProfileDerivation{}, fmt.Errorf("model recipe: decode profile derivation: %w", err)
	}
	document := CatalogProfileDerivation{catalogProfileDerivationBody: body}
	if err := document.validateShape(); err != nil {
		return CatalogProfileDerivation{}, err
	}
	canonical, err := catalogProfileDerivationContent(document)
	if err != nil {
		return CatalogProfileDerivation{}, err
	}
	if !bytes.Equal(canonical, content) {
		return CatalogProfileDerivation{}, errors.New("model recipe: non-canonical profile derivation")
	}
	document.ID, err = catalogProfileDerivationContract.Identify(canonical)
	return document, err
}

func (d CatalogProfileDerivation) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindEvidence {
		return errors.New("model recipe: invalid profile derivation identity")
	}
	if err := d.validateShape(); err != nil {
		return err
	}
	content, err := catalogProfileDerivationContent(d)
	if err != nil {
		return err
	}
	if err := catalogProfileDerivationContract.ValidateIdentity(d.ID, content); err != nil {
		return errors.New("model recipe: profile derivation identity mismatch")
	}
	return nil
}

func (d CatalogProfileDerivation) Content() (artifact.Content, error) {
	if err := d.ValidateIdentity(); err != nil {
		return artifact.Content{}, err
	}
	content, err := catalogProfileDerivationContent(d)
	if err != nil {
		return artifact.Content{}, err
	}
	return catalogProfileDerivationContract.Content(d.ID, content)
}

func (d CatalogProfileDerivation) validateShape() error {
	if d.Version != CatalogProfileDerivationVersion || d.ProfileVersion != ProfileVersion ||
		d.Architecture == "" || d.Source != catalogProfileSource ||
		!validSHA256(d.FactSchemaSHA256) || !validSHA256(d.PolicySHA256) {
		return errors.New("model recipe: invalid profile derivation")
	}
	return nil
}

func catalogProfileDerivationContent(document CatalogProfileDerivation) ([]byte, error) {
	content, err := json.Marshal(document.catalogProfileDerivationBody)
	if err != nil {
		return nil, fmt.Errorf("model recipe: encode profile derivation: %w", err)
	}
	return content, nil
}

func sha256Hex(content []byte) string {
	digest := sha256.Sum256(content)
	return hex.EncodeToString(digest[:])
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func profilePublicationFacts(document ProfileDocument) (
	[]artifact.Content,
	[]artifact.Lineage,
	error,
) {
	profileContent, err := ProfileContent(document)
	if err != nil {
		return nil, nil, err
	}
	contents := []artifact.Content{profileContent}
	if document.Version == LegacyProfileVersion {
		return contents, nil, nil
	}
	catalog, err := NewCatalogProfileDerivation(document.Policy)
	if err != nil {
		return nil, nil, err
	}
	seen := make(map[artifact.ID]struct{})
	lineage := make([]artifact.Lineage, 0, len(document.Provenance))
	for _, provenance := range document.Provenance {
		catalogClaim := provenance.Origin == ProfileFactConfig &&
			strings.HasPrefix(provenance.SourceField, "architecture_catalog.")
		if catalogClaim && provenance.DerivationID != catalog.ID {
			return nil, nil, errors.New("model recipe: catalog provenance differs from derivation evidence")
		}
		if _, duplicate := seen[provenance.DerivationID]; duplicate {
			continue
		}
		seen[provenance.DerivationID] = struct{}{}
		lineage = append(lineage, artifact.Lineage{
			Child: document.ID, Parent: provenance.DerivationID,
			Relation: artifact.RelationDerivedFrom,
		})
		if provenance.DerivationID == catalog.ID {
			content, err := catalog.Content()
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, content)
		}
	}
	return contents, lineage, nil
}

func validateStoredProfileProvenance(
	ctx context.Context,
	store artifact.Reader,
	document ProfileDocument,
) error {
	if document.Version == LegacyProfileVersion {
		return nil
	}
	parents, err := store.Parents(ctx, document.ID)
	if err != nil {
		return err
	}
	derivedFrom := make(map[artifact.ID]struct{}, len(parents))
	for _, parent := range parents {
		if parent.Relation == artifact.RelationDerivedFrom {
			derivedFrom[parent.Parent] = struct{}{}
		}
	}
	catalog, err := NewCatalogProfileDerivation(document.Policy)
	if err != nil {
		return err
	}
	checked := make(map[artifact.ID]struct{}, len(document.Provenance))
	for _, provenance := range document.Provenance {
		if _, ok := derivedFrom[provenance.DerivationID]; !ok {
			return errors.New("model recipe: profile derivation lineage is absent")
		}
		catalogClaim := provenance.Origin == ProfileFactConfig &&
			strings.HasPrefix(provenance.SourceField, "architecture_catalog.")
		if catalogClaim && provenance.DerivationID != catalog.ID {
			return errors.New("model recipe: stored catalog provenance differs from derivation evidence")
		}
		if _, ok := checked[provenance.DerivationID]; ok {
			continue
		}
		checked[provenance.DerivationID] = struct{}{}
		descriptor, ok, err := store.Artifact(ctx, provenance.DerivationID)
		if err != nil {
			return err
		}
		if !ok || descriptor.ID.Kind() != artifact.KindEvidence {
			return errors.New("model recipe: profile derivation evidence is absent")
		}
	}
	return nil
}
