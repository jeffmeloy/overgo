// Package finding owns the store's findings surface: the live adversarial
// register skill.md Evidence Standards names. A finding is a typed, immutable,
// content-addressed document with an owner surface, evidence lineage, a
// closure path, and a failable check -- findings persist while live, close
// only with an implemented fix plus failable check, refute only with direct
// counter-evidence.
package finding

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	Version   uint16 = 1
	MediaType        = "application/vnd.overgo.finding+json"
	Schema           = "overgo/finding/v1"
)

type Severity string

const (
	SeverityLow    Severity = "low"
	SeverityMedium Severity = "medium"
	SeverityHigh   Severity = "high"
)

type Status string

const (
	StatusOpen    Status = "open"
	StatusClosed  Status = "closed"
	StatusRefuted Status = "refuted"
)

var contract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: MediaType, Schema: Schema,
}

var codec = artifact.DocumentCodec[Document]{
	Name: "finding", Contract: contract,
	Decode: func(data []byte, value *Document) error { return strictjson.DecodeBytes(data, value) },
	Encode: func(document Document) ([]byte, error) {
		document.ID = artifact.ID{}
		return json.Marshal(document)
	},
	Canonicalize: canonicalize,
	Clone: func(document Document) Document {
		document.OwnerSurfaces = slices.Clone(document.OwnerSurfaces)
		document.Evidence = slices.Clone(document.Evidence)
		return document
	},
	Identity:    func(document Document) artifact.ID { return document.ID },
	SetIdentity: func(document *Document, id artifact.ID) { document.ID = id },
}

// Document: one live finding.
type Document struct {
	Version       uint16        `json:"version"`
	Title         string        `json:"title"`
	Severity      Severity      `json:"severity"`
	Status        Status        `json:"status"`
	OwnerSurfaces []artifact.ID `json:"owner_surfaces"`
	Evidence      []artifact.ID `json:"evidence"`
	ClosurePath   string        `json:"closure_path"`
	FailableCheck string        `json:"failable_check"`
	ID            artifact.ID   `json:"-"`
}

func New(
	title string,
	severity Severity,
	status Status,
	ownerSurfaces, evidence []artifact.ID,
	closurePath, failableCheck string,
) (Document, error) {
	return codec.New(Document{
		Version: Version, Title: title, Severity: severity, Status: status,
		OwnerSurfaces: slices.Clone(ownerSurfaces), Evidence: slices.Clone(evidence),
		ClosurePath: closurePath, FailableCheck: failableCheck,
	})
}

func Parse(data []byte) (Document, error)             { return codec.Parse(data) }
func Normalize(data []byte) (Document, []byte, error) { return codec.Normalize(data) }
func (d Document) Content() (artifact.Content, error) { return codec.Content(d) }
func (d Document) ValidateIdentity() error            { return codec.ValidateIdentity(d) }

func (d Document) Lineage() []artifact.Lineage {
	parents := append(slices.Clone(d.OwnerSurfaces), d.Evidence...)
	edges := make([]artifact.Lineage, len(parents))
	for index, parent := range parents {
		edges[index] = artifact.Lineage{Child: d.ID, Parent: parent, Relation: artifact.RelationDependsOn}
	}
	return edges
}

func (d Document) Batch(key string) (artifact.Batch, error) {
	return codec.Batch(key, d, d.Lineage(), nil)
}

// NewTextBatch packages an agent-authored finding and its textual owner and
// evidence anchors into one atomic store commit. The typed document remains
// the authority; text inputs are content-addressed parents, not a side ledger.
func NewTextBatch(title string, severity Severity, ownerSurfaces, evidence []string, closurePath, failableCheck string) (Document, artifact.Batch, error) {
	owners, ownerContents, err := textContents(artifact.KindFile, ownerSurfaces)
	if err != nil {
		return Document{}, artifact.Batch{}, err
	}
	evidenceIDs, evidenceContents, err := textContents(artifact.KindEvidence, evidence)
	if err != nil {
		return Document{}, artifact.Batch{}, err
	}
	document, err := New(title, severity, StatusOpen, owners, evidenceIDs, closurePath, failableCheck)
	if err != nil {
		return Document{}, artifact.Batch{}, err
	}
	content, err := document.Content()
	if err != nil {
		return Document{}, artifact.Batch{}, err
	}
	contents := append(append(ownerContents, evidenceContents...), content)
	batch, err := artifact.NewDocumentBatch("finding/"+document.ID.String(), contents, document.Lineage(), nil)
	return document, batch, err
}

func textContents(kind artifact.Kind, values []string) ([]artifact.ID, []artifact.Content, error) {
	ids := make([]artifact.ID, 0, len(values))
	contents := make([]artifact.Content, 0, len(values))
	seen := make(map[artifact.ID]bool, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if !textcheck.Bounded(value, 4096, "\x00\r") {
			return nil, nil, errors.New("finding: invalid text anchor")
		}
		data := []byte(value)
		id, err := artifact.IdentifyBytes(kind, data)
		if err != nil {
			return nil, nil, err
		}
		if seen[id] {
			continue
		}
		seen[id] = true
		ids = append(ids, id)
		contents = append(contents, artifact.Content{Descriptor: artifact.Descriptor{
			ID: id, Size: uint64(len(data)), MediaType: "text/plain",
		}, Data: data})
	}
	return ids, contents, nil
}

func canonicalize(document *Document) error {
	if document == nil || document.Version != Version ||
		!textcheck.Bounded(document.Title, 4096, "\x00\r") || !textcheck.Bounded(document.ClosurePath, 4096, "\x00\r") ||
		!textcheck.Bounded(document.FailableCheck, 4096, "\x00\r") ||
		len(document.OwnerSurfaces) == 0 || len(document.Evidence) == 0 {
		return errors.New("finding: invalid document")
	}
	switch document.Severity {
	case SeverityLow, SeverityMedium, SeverityHigh:
	default:
		return errors.New("finding: invalid severity")
	}
	switch document.Status {
	case StatusOpen, StatusClosed, StatusRefuted:
	default:
		return errors.New("finding: invalid status")
	}
	for _, list := range [][]artifact.ID{document.OwnerSurfaces, document.Evidence} {
		for _, id := range list {
			if !id.Valid() {
				return errors.New("finding: invalid reference")
			}
		}
	}
	sortIDs(document.OwnerSurfaces)
	sortIDs(document.Evidence)
	return nil
}

func sortIDs(ids []artifact.ID) {
	slices.SortFunc(ids, func(a, b artifact.ID) int {
		return strings.Compare(a.String(), b.String())
	})
}
