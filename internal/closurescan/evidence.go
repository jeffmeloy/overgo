package closurescan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
)

const (
	CensusEvidenceMediaType = "application/vnd.overgo.magic-census+json"
	CensusEvidenceSchema    = "overgo/magic-census-evidence/v2"
	CensusEvidenceAlias     = "closure/census/latest"
)

type ClosurePressure struct {
	ActiveDocuments int `json:"active_documents"`
	OpenDocuments   int `json:"open_documents"`
	StaleBindings   int `json:"stale_bindings"`
}

type UnresolvedClosure struct {
	Document artifact.ID                   `json:"document"`
	Name     string                        `json:"name"`
	Tier     closureledger.Tier            `json:"tier"`
	Bindings []closureledger.SourceBinding `json:"bindings"`
}

type CensusEvidence struct {
	Source          string              `json:"source"`
	CatalogHead     string              `json:"catalog_head"`
	CatalogSequence uint64              `json:"catalog_sequence"`
	Counts          CensusCounts        `json:"counts"`
	Owners          []OwnerPressure     `json:"owners"`
	Files           []FilePressure      `json:"files"`
	Pressure        ClosurePressure     `json:"closure_pressure"`
	Unresolved      []UnresolvedClosure `json:"unresolved"`
	Stale           []BindingIssue      `json:"stale,omitempty"`
	ID              artifact.ID         `json:"-"`
}

var censusEvidenceCodec = artifact.JSONDocumentCodec(
	"magic census evidence", artifact.KindEvidence, CensusEvidenceMediaType, CensusEvidenceSchema,
	validateCensusEvidence, func(value CensusEvidence) artifact.ID { return value.ID },
	func(value *CensusEvidence, id artifact.ID) { value.ID = id }, cloneCensusEvidence,
)

func NewCensusEvidence(census Census, head artifact.CommitID, sequence uint64, active []closureledger.Document, stale []BindingIssue) (CensusEvidence, error) {
	evidence := CensusEvidence{
		Source: census.Source, CatalogHead: head.String(), CatalogSequence: sequence,
		Counts: census.Counts, Owners: census.Owners, Files: census.Files,
		Pressure: ClosurePressure{ActiveDocuments: len(active), StaleBindings: len(stale)}, Stale: stale,
	}
	for _, document := range active {
		if document.Status != closureledger.StatusOpen {
			continue
		}
		evidence.Unresolved = append(evidence.Unresolved, UnresolvedClosure{
			Document: document.ID, Name: document.Name, Tier: document.Tier,
			Bindings: document.Bindings,
		})
	}
	evidence.Pressure.OpenDocuments = len(evidence.Unresolved)
	return censusEvidenceCodec.New(evidence)
}

func ReadCensusEvidence(ctx context.Context, reader artifact.Reader, id artifact.ID) (CensusEvidence, bool, error) {
	return censusEvidenceCodec.Read(ctx, reader, id)
}

// ParseCensusEvidence validates stored census bytes.
func ParseCensusEvidence(data []byte) (CensusEvidence, error) {
	return censusEvidenceCodec.Parse(data)
}

func (e CensusEvidence) Batch(previous *artifact.ID) (artifact.Batch, error) {
	alias := artifact.AliasBinding{Name: CensusEvidenceAlias, Target: e.ID, Previous: previous}
	lineage := make([]artifact.Lineage, len(e.Unresolved))
	for index, row := range e.Unresolved {
		lineage[index] = artifact.Lineage{Child: e.ID, Parent: row.Document, Relation: artifact.RelationDependsOn}
	}
	return censusEvidenceCodec.Batch("closure/census/"+e.ID.String(), e, lineage, []artifact.AliasBinding{alias})
}

func validateCensusEvidence(value *CensusEvidence) error {
	if value == nil || !digest(value.Source) || !digest(value.CatalogHead) || value.Pressure.ActiveDocuments < value.Pressure.OpenDocuments {
		return errors.New("closure scan: invalid census evidence")
	}
	var named, inline, assumptions, policy, groups, sites int
	seenOwners := map[string]bool{}
	for _, owner := range value.Owners {
		if owner.Package == "" || seenOwners[owner.Package] {
			return errors.New("closure scan: invalid census owner")
		}
		seenOwners[owner.Package] = true
		named += owner.NamedConstants
		inline += owner.InlineLiterals
		assumptions += owner.AssumptionHints
		policy += owner.TestPolicyCopies
		groups += owner.RepeatedGroups
		sites += owner.RepeatedSites
	}
	if named != value.Counts.NamedConstants || inline != value.Counts.InlineLiterals || assumptions != value.Counts.AssumptionHints ||
		policy != value.Counts.TestPolicyCopies || groups != value.Counts.RepeatedGroups || sites != value.Counts.RepeatedSites ||
		len(value.Unresolved) != value.Pressure.OpenDocuments || len(value.Stale) != value.Pressure.StaleBindings {
		return errors.New("closure scan: inconsistent census evidence")
	}
	if err := validateFilePressure(value.Files, value.Counts); err != nil {
		return err
	}
	for _, row := range value.Unresolved {
		hasBinding := false
		for range row.Bindings {
			hasBinding = true
			break
		}
		if !row.Document.Valid() || row.Name == "" || row.Tier != closureledger.TierDerivationBlocked || !hasBinding {
			return errors.New("closure scan: invalid unresolved closure")
		}
	}
	return nil
}

func digest(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func cloneCensusEvidence(value CensusEvidence) CensusEvidence {
	value.Owners = slices.Clone(value.Owners)
	value.Files = slices.Clone(value.Files)
	value.Unresolved = slices.Clone(value.Unresolved)
	value.Stale = slices.Clone(value.Stale)
	for index := range value.Unresolved {
		value.Unresolved[index].Bindings = slices.Clone(value.Unresolved[index].Bindings)
	}
	return value
}

func validateFilePressure(files []FilePressure, counts CensusCounts) error {
	var production, tests, named, inline, assumptions, testLiterals, fixtures, assertions, policy int
	seen := make(map[string]bool, len(files))
	var previous FilePressure
	hasPrevious := false
	for _, file := range files {
		if file.File == "" || seen[file.File] || file.DecisionSurfaces != file.NamedConstants+file.InlineLiterals+file.AssumptionHints+file.TestPolicyCopies {
			return errors.New("closure scan: invalid file pressure")
		}
		if hasPrevious && lessFilePressure(previous, file) {
			return errors.New("closure scan: unordered file pressure")
		}
		previous, hasPrevious = file, true
		seen[file.File] = true
		if file.Test {
			tests++
		} else {
			production++
		}
		named += file.NamedConstants
		inline += file.InlineLiterals
		assumptions += file.AssumptionHints
		testLiterals += file.TestLiterals
		fixtures += file.TestFixtures
		assertions += file.TestAssertions
		policy += file.TestPolicyCopies
	}
	if production != counts.ProductionFiles || tests != counts.TestFiles || named != counts.NamedConstants ||
		inline != counts.InlineLiterals || assumptions != counts.AssumptionHints || testLiterals != counts.TestLiterals ||
		fixtures != counts.TestFixtures || assertions != counts.TestAssertions || policy != counts.TestPolicyCopies {
		return errors.New("closure scan: inconsistent file pressure")
	}
	return nil
}

func lessFilePressure(left, right FilePressure) bool {
	if left.DecisionSurfaces != right.DecisionSurfaces {
		return left.DecisionSurfaces < right.DecisionSurfaces
	}
	if left.RepeatedGroups != right.RepeatedGroups {
		return left.RepeatedGroups < right.RepeatedGroups
	}
	return left.File > right.File
}
