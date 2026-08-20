package closureledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"path"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	Version   uint16 = 3
	MediaType        = "application/vnd.overgo.closure-ledger+json"
	Schema           = "overgo/closure-ledger/v3"

	maxNameBytes  = 256
	maxTextBytes  = 32 << 10
	maxValueBytes = 16 << 10
)

var documentContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: MediaType, Schema: Schema,
}

var documentCodec = artifact.JSONDocumentCodec(
	"closure ledger", documentContract.Kind, documentContract.MediaType, documentContract.Schema,
	canonicalize, func(value Document) artifact.ID { return value.ID },
	func(value *Document, id artifact.ID) { value.ID = id }, cloneDocument,
)

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

type BindingKind string

const (
	BindingConstant   BindingKind = "constant"
	BindingLiteral    BindingKind = "literal"
	BindingAssumption BindingKind = "assumption"
)

type SourceBinding struct {
	Kind       BindingKind `json:"kind"`
	Package    string      `json:"package"`
	File       string      `json:"file"`
	Scope      string      `json:"scope"`
	Name       string      `json:"name"`
	Line       int         `json:"line"`
	Expression string      `json:"expression"`
	SourceID   string      `json:"source_id"`
	CallsiteID string      `json:"callsite_id"`
	Owner      artifact.ID `json:"owner"`
}

// Document: immutable closure obligation.
type Document struct {
	Version       uint16          `json:"version"`
	Name          string          `json:"name"`
	Value         json.RawMessage `json:"value"`
	Tier          Tier            `json:"tier"`
	Status        Status          `json:"status"`
	Understanding string          `json:"understanding"`
	Bindings      []SourceBinding `json:"bindings"`
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
	understanding string,
	bindings []SourceBinding,
	closurePath, rerankTrigger string,
	fixture artifact.ID,
) (Document, error) {
	document := Document{
		Version: Version, Name: name, Value: slices.Clone(value), Tier: tier, Status: status,
		Understanding: understanding, Bindings: slices.Clone(bindings), ClosurePath: closurePath,
		RerankTrigger: rerankTrigger, Fixture: fixture,
	}
	return documentCodec.New(document)
}

func Parse(content []byte) (Document, error) {
	return documentCodec.Parse(content)
}

func (d Document) ValidateIdentity() error {
	return documentCodec.ValidateIdentity(d)
}

func (d Document) Content() (artifact.Content, error) {
	return documentCodec.Content(d)
}

func (d Document) Lineage() []artifact.Lineage {
	edges := make([]artifact.Lineage, 0, len(d.Bindings)+1)
	seen := map[artifact.ID]bool{}
	for _, binding := range d.Bindings {
		if seen[binding.Owner] {
			continue
		}
		seen[binding.Owner] = true
		edges = append(edges, artifact.Lineage{
			Child: d.ID, Parent: binding.Owner, Relation: artifact.RelationDependsOn,
		})
	}
	edges = append(edges, artifact.Lineage{
		Child: d.ID, Parent: d.Fixture, Relation: artifact.RelationDependsOn,
	})
	return edges
}

func (d Document) Batch(key string, alias *artifact.AliasBinding) (artifact.Batch, error) {
	var aliases []artifact.AliasBinding
	if alias != nil {
		if alias.Target != d.ID {
			return artifact.Batch{}, errors.New("closure ledger: alias targets another document")
		}
		aliases = append(aliases, *alias)
	}
	return documentCodec.Batch(key, d, d.Lineage(), aliases)
}

func canonicalize(document *Document) error {
	if document == nil || document.Version != Version || !validName(document.Name) ||
		!validTier(document.Tier) || !validTierStatus(document.Tier, document.Status) ||
		strings.TrimSpace(document.Understanding) == "" ||
		strings.TrimSpace(document.ClosurePath) == "" || strings.TrimSpace(document.RerankTrigger) == "" ||
		!textcheck.Bounded(document.Understanding, maxTextBytes, "\x00\r") ||
		!textcheck.Bounded(document.ClosurePath, maxTextBytes, "\x00\r") ||
		!textcheck.Bounded(document.RerankTrigger, maxTextBytes, "\x00\r") ||
		!document.Fixture.Valid() || len(document.Bindings) == 0 {
		return errors.New("closure ledger: invalid document")
	}
	value, err := canonicalValue(document.Value)
	if err != nil {
		return err
	}
	document.Value = value
	sort.Slice(document.Bindings, func(i, j int) bool {
		return bindingKey(document.Bindings[i]) < bindingKey(document.Bindings[j])
	})
	for index, binding := range document.Bindings {
		if !validBinding(binding) || index > 0 && bindingKey(document.Bindings[index-1]) == bindingKey(binding) {
			return errors.New("closure ledger: invalid source binding")
		}
	}
	return nil
}

func validBinding(binding SourceBinding) bool {
	if binding.Kind != BindingConstant && binding.Kind != BindingLiteral && binding.Kind != BindingAssumption ||
		!validName(binding.Name) || binding.Package == "" || binding.Scope == "" || binding.Line <= 0 ||
		!textcheck.Bounded(binding.Expression, maxTextBytes, "\x00\r") || !binding.Owner.Valid() {
		return false
	}
	clean := path.Clean(binding.File)
	if clean != binding.File || clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return false
	}
	source, sourceErr := hex.DecodeString(binding.SourceID)
	callsites, callsiteErr := hex.DecodeString(binding.CallsiteID)
	return sourceErr == nil && len(source) == sha256.Size && callsiteErr == nil && len(callsites) == sha256.Size
}

func bindingKey(binding SourceBinding) string {
	return bindingDeclarationKey(binding) + "\x00" + binding.SourceID + "\x00" + binding.CallsiteID
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

func cloneDocument(document Document) Document {
	document.Value = slices.Clone(document.Value)
	document.Bindings = slices.Clone(document.Bindings)
	return document
}
