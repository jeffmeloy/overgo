package closureledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	Version   uint16 = 1
	MediaType        = "application/vnd.overgo.closure-ledger+json"
	Schema           = "overgo/closure-ledger/v1"

	maxNameBytes  = 256
	maxTextBytes  = 32 << 10
	maxValueBytes = 16 << 10
)

var documentContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: MediaType, Schema: Schema,
}

type Tier string

const (
	TierDerivationBlocked Tier = "derivation-blocked"
	TierPhenomenology     Tier = "phenomenology-validated"
	TierMeasuredNull      Tier = "closed-by-measured-null"
	TierAdaptationShipped Tier = "adaptation-shipped"
	TierImplementation    Tier = "implementation-constraint"
	TierMathematicalFact  Tier = "mathematical-fact"
)

type Status string

const (
	StatusOpen      Status = "open"
	StatusMonitored Status = "monitored"
	StatusClosed    Status = "closed"
)

// Document: immutable closure obligation.
type Document struct {
	Version       uint16          `json:"version"`
	Name          string          `json:"name"`
	Value         json.RawMessage `json:"value"`
	Tier          Tier            `json:"tier"`
	Status        Status          `json:"status"`
	OwnerSurfaces []artifact.ID   `json:"owner_surfaces"`
	ClosurePath   string          `json:"closure_path"`
	RerankTrigger string          `json:"rerank_trigger"`
	Fixture       artifact.ID     `json:"pinning_fixture"`
	ID            artifact.ID     `json:"-"`
}

func New(
	name string,
	value json.RawMessage,
	tier Tier,
	status Status,
	ownerSurfaces []artifact.ID,
	closurePath, rerankTrigger string,
	fixture artifact.ID,
) (Document, error) {
	document := Document{
		Version: Version, Name: name, Value: slices.Clone(value), Tier: tier, Status: status,
		OwnerSurfaces: slices.Clone(ownerSurfaces), ClosurePath: closurePath,
		RerankTrigger: rerankTrigger, Fixture: fixture,
	}
	if err := canonicalize(&document); err != nil {
		return Document{}, err
	}
	content, err := documentContent(document)
	if err != nil {
		return Document{}, err
	}
	document.ID, err = documentContract.Identify(content)
	return document, err
}

func Parse(content []byte) (Document, error) {
	var body Document
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return Document{}, fmt.Errorf("closure ledger: decode document: %w", err)
	}
	document, err := New(
		body.Name, body.Value, body.Tier, body.Status, body.OwnerSurfaces,
		body.ClosurePath, body.RerankTrigger, body.Fixture,
	)
	if err != nil {
		return Document{}, err
	}
	canonical, err := document.ContentBytes()
	if err != nil {
		return Document{}, err
	}
	if !bytes.Equal(canonical, content) {
		return Document{}, errors.New("closure ledger: non-canonical document")
	}
	return document, nil
}

func (d Document) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindEvidence {
		return errors.New("closure ledger: invalid identity")
	}
	canonical := cloneDocument(d)
	canonical.ID = artifact.ID{}
	if err := canonicalize(&canonical); err != nil {
		return err
	}
	canonical.ID = d.ID
	if !sameDocument(d, canonical) {
		return errors.New("closure ledger: document is not canonical")
	}
	canonical.ID = artifact.ID{}
	content, err := documentContent(canonical)
	if err != nil {
		return err
	}
	if err := documentContract.ValidateIdentity(d.ID, content); err != nil {
		return errors.New("closure ledger: identity mismatch")
	}
	return nil
}

func (d Document) ContentBytes() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	d.ID = artifact.ID{}
	return documentContent(d)
}

func (d Document) Content() (artifact.Content, error) {
	id := d.ID
	content, err := d.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return documentContract.Content(id, content)
}

func (d Document) Lineage() []artifact.Lineage {
	edges := make([]artifact.Lineage, 0, len(d.OwnerSurfaces)+1)
	for _, owner := range d.OwnerSurfaces {
		edges = append(edges, artifact.Lineage{
			Child: d.ID, Parent: owner, Relation: artifact.RelationDependsOn,
		})
	}
	edges = append(edges, artifact.Lineage{
		Child: d.ID, Parent: d.Fixture, Relation: artifact.RelationDependsOn,
	})
	return edges
}

func (d Document) Batch(key string, alias *artifact.AliasBinding) (artifact.Batch, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	var aliases []artifact.AliasBinding
	if alias != nil {
		if alias.Target != d.ID {
			return artifact.Batch{}, errors.New("closure ledger: alias targets another document")
		}
		aliases = append(aliases, *alias)
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, d.Lineage(), aliases)
}

func canonicalize(document *Document) error {
	if document == nil || document.Version != Version || !validName(document.Name) ||
		!validTier(document.Tier) || !validTierStatus(document.Tier, document.Status) ||
		!validText(document.ClosurePath) || !validText(document.RerankTrigger) ||
		!document.Fixture.Valid() || len(document.OwnerSurfaces) == 0 {
		return errors.New("closure ledger: invalid document")
	}
	value, err := canonicalValue(document.Value)
	if err != nil {
		return err
	}
	document.Value = value
	sort.Slice(document.OwnerSurfaces, func(i, j int) bool {
		return document.OwnerSurfaces[i].String() < document.OwnerSurfaces[j].String()
	})
	for index, owner := range document.OwnerSurfaces {
		if !owner.Valid() || index > 0 && document.OwnerSurfaces[index-1] == owner {
			return errors.New("closure ledger: invalid owner surface")
		}
	}
	return nil
}

func canonicalValue(value json.RawMessage) (json.RawMessage, error) {
	if len(value) == 0 || len(value) > maxValueBytes {
		return nil, errors.New("closure ledger: invalid value")
	}
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()
	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return nil, errors.New("closure ledger: invalid value")
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return nil, errors.New("closure ledger: invalid value")
	}
	canonical, err := json.Marshal(decoded)
	if err != nil || len(canonical) > maxValueBytes {
		return nil, errors.New("closure ledger: invalid value")
	}
	return canonical, nil
}

func validTier(tier Tier) bool {
	switch tier {
	case TierDerivationBlocked, TierPhenomenology, TierMeasuredNull,
		TierAdaptationShipped, TierImplementation, TierMathematicalFact:
		return true
	default:
		return false
	}
}

func validTierStatus(tier Tier, status Status) bool {
	switch tier {
	case TierDerivationBlocked:
		return status == StatusOpen
	case TierPhenomenology:
		return status == StatusMonitored
	case TierAdaptationShipped:
		return status == StatusMonitored || status == StatusClosed
	case TierMeasuredNull, TierImplementation, TierMathematicalFact:
		return status == StatusClosed
	default:
		return false
	}
}

func validName(value string) bool {
	if value == "" || len(value) > maxNameBytes || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || character == '.' || character == '-' || character == '_' {
			continue
		}
		return false
	}
	return true
}

func validText(value string) bool {
	return value != "" && len(value) <= maxTextBytes && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\x00\r")
}

func documentContent(document Document) ([]byte, error) {
	document.ID = artifact.ID{}
	content, err := json.Marshal(document)
	if err != nil {
		return nil, fmt.Errorf("closure ledger: encode document: %w", err)
	}
	return content, nil
}

func cloneDocument(document Document) Document {
	document.Value = slices.Clone(document.Value)
	document.OwnerSurfaces = slices.Clone(document.OwnerSurfaces)
	return document
}

func sameDocument(left, right Document) bool {
	return left.Version == right.Version && left.Name == right.Name && bytes.Equal(left.Value, right.Value) &&
		left.Tier == right.Tier && left.Status == right.Status &&
		slices.Equal(left.OwnerSurfaces, right.OwnerSurfaces) &&
		left.ClosurePath == right.ClosurePath && left.RerankTrigger == right.RerankTrigger &&
		left.Fixture == right.Fixture
}
