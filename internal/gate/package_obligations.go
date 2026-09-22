package gate

import (
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// One landing's package obligations are one document. The obligation each
// package receipt is keyed by stays content-addressed and computable from the
// package's invocation, mode and sources, so the receipt alias and the reuse
// check never needed the obligation stored as a record of its own; the gate
// wrote one such record per package per landing, about 127 a landing, the
// store's largest schema. The document lists them instead: one record per
// preparation, the same identity whenever the same packages meet the same
// inputs in the same environment.
const (
	packageObligationsSchema    = "overgo/gate-package-obligations/v1"
	packageObligationsMediaType = "application/vnd.overgo.gate-package-obligations+json"
	packageObligationsAlias     = "gate/package-obligations/"
)

// packageObligationEntry names one package's obligation and what it binds.
type packageObligationEntry struct {
	Package    string      `json:"package"`
	Mode       string      `json:"mode"`
	Task       artifact.ID `json:"task"`
	Input      artifact.ID `json:"input"`
	Obligation artifact.ID `json:"obligation"`
}

// packageObligations is the per-preparation list, ordered by package so the
// same set always has the same identity.
type packageObligations struct {
	Version     uint16                   `json:"version"`
	ID          artifact.ID              `json:"-"`
	Environment artifact.ID              `json:"environment"`
	Entries     []packageObligationEntry `json:"entries"`
}

var packageObligationsCodec = artifact.JSONDocumentCodec(
	"gate package obligations", artifact.KindEvidence, packageObligationsMediaType, packageObligationsSchema,
	func(value *packageObligations) error {
		if value.Version != artifact.InitialDocumentVersion || value.Environment.Kind() != artifact.KindEvidence || len(value.Entries) == 0 {
			return errors.New("gate package obligations: invalid binding")
		}
		if !slices.IsSortedFunc(value.Entries, func(a, b packageObligationEntry) int { return strings.Compare(a.Package, b.Package) }) {
			return errors.New("gate package obligations: entries must be ordered by package")
		}
		for index, entry := range value.Entries {
			if strings.TrimSpace(entry.Package) == "" || strings.TrimSpace(entry.Mode) == "" ||
				!entry.Task.Valid() || !entry.Input.Valid() || entry.Obligation.Kind() != artifact.KindEvidence ||
				index > 0 && value.Entries[index-1].Package == entry.Package {
				return errors.New("gate package obligations: invalid or duplicate entry")
			}
		}
		return nil
	}, func(value packageObligations) artifact.ID { return value.ID },
	func(value *packageObligations, id artifact.ID) { value.ID = id },
	func(value packageObligations) packageObligations {
		value.Entries = slices.Clone(value.Entries)
		return value
	},
)
