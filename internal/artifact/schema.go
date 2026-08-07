package artifact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	maxAliasBytes    = 512
	maxBatchKeyBytes = 512
)

// Descriptor: immutable content fact
type Descriptor struct {
	ID        ID     `json:"id"`
	Size      uint64 `json:"size"`
	MediaType string `json:"media_type,omitempty"`
	Schema    string `json:"schema,omitempty"`
}

func (d Descriptor) Validate() error {
	if !d.ID.Valid() {
		return errors.New("artifact: descriptor has invalid ID")
	}
	if strings.TrimSpace(d.MediaType) != d.MediaType || strings.ContainsAny(d.MediaType, "\r\n") {
		return errors.New("artifact: descriptor has invalid media type")
	}
	if strings.TrimSpace(d.Schema) != d.Schema || strings.ContainsAny(d.Schema, "\r\n") {
		return errors.New("artifact: descriptor has invalid schema")
	}
	return nil
}

// Relation: child-to-parent lineage role
type Relation uint8

const (
	RelationInvalid Relation = iota
	RelationDerivedFrom
	RelationContains
	RelationConvertedFrom
	RelationQuantizedFrom
	RelationTrainedFrom
	RelationAdaptedFrom
	RelationTokenizedBy
	RelationProjectedBy
	RelationProducedBy
	RelationDependsOn
	RelationDuplicateOf
)

var relationNames = [...]string{
	RelationInvalid:       "invalid",
	RelationDerivedFrom:   "derived-from",
	RelationContains:      "contains",
	RelationConvertedFrom: "converted-from",
	RelationQuantizedFrom: "quantized-from",
	RelationTrainedFrom:   "trained-from",
	RelationAdaptedFrom:   "adapted-from",
	RelationTokenizedBy:   "tokenized-by",
	RelationProjectedBy:   "projected-by",
	RelationProducedBy:    "produced-by",
	RelationDependsOn:     "depends-on",
	RelationDuplicateOf:   "duplicate-of",
}

func (r Relation) String() string {
	if int(r) >= len(relationNames) {
		return relationNames[RelationInvalid]
	}
	return relationNames[r]
}

func ParseRelation(value string) (Relation, error) {
	for relation, name := range relationNames {
		if relation > 0 && value == name {
			return Relation(relation), nil
		}
	}
	return RelationInvalid, fmt.Errorf("artifact: unknown relation %q", value)
}

func (r Relation) MarshalText() ([]byte, error) {
	if r == RelationInvalid || int(r) >= len(relationNames) {
		return nil, errors.New("artifact: invalid relation")
	}
	return []byte(r.String()), nil
}

func (r Relation) MarshalJSON() ([]byte, error) {
	text, err := r.MarshalText()
	if err != nil {
		return nil, err
	}
	return json.Marshal(string(text))
}

func (r *Relation) UnmarshalText(data []byte) error {
	if r == nil {
		return errors.New("artifact: nil relation target")
	}
	parsed, err := ParseRelation(string(data))
	if err != nil {
		return err
	}
	*r = parsed
	return nil
}

func (r *Relation) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("artifact: decode relation: %w", err)
	}
	return r.UnmarshalText([]byte(value))
}

// Lineage: immutable dependency edge
type Lineage struct {
	Child    ID       `json:"child"`
	Parent   ID       `json:"parent"`
	Relation Relation `json:"relation"`
}

func (l Lineage) Validate() error {
	if !l.Child.Valid() || !l.Parent.Valid() {
		return errors.New("artifact: lineage has invalid endpoint")
	}
	if l.Child == l.Parent {
		return errors.New("artifact: lineage self-cycle")
	}
	if l.Relation == RelationInvalid || int(l.Relation) >= len(relationNames) {
		return errors.New("artifact: lineage has invalid relation")
	}
	return nil
}

// AliasBinding: compare-and-set mutable name; nil Previous requires unbound name
type AliasBinding struct {
	Name     string `json:"name"`
	Target   ID     `json:"target"`
	Previous *ID    `json:"previous,omitempty"`
}

func (b AliasBinding) Validate() error {
	if b.Name == "" || len(b.Name) > maxAliasBytes || strings.TrimSpace(b.Name) != b.Name || strings.ContainsAny(b.Name, "\r\n") {
		return errors.New("artifact: invalid alias")
	}
	if !b.Target.Valid() {
		return errors.New("artifact: alias has invalid target")
	}
	if b.Previous != nil && !b.Previous.Valid() {
		return errors.New("artifact: alias has invalid previous target")
	}
	return nil
}

// Batch: one atomic repository commit
type Batch struct {
	Key       string          `json:"key"`
	Artifacts []Descriptor    `json:"artifacts,omitempty"`
	Contents  []Content       `json:"contents,omitempty"`
	Manifests []Manifest      `json:"manifests,omitempty"`
	Lineage   []Lineage       `json:"lineage,omitempty"`
	Aliases   []AliasBinding  `json:"aliases,omitempty"`
	Locations []LocationEvent `json:"locations,omitempty"`
}

func (b Batch) Validate() error {
	if b.Key == "" || len(b.Key) > maxBatchKeyBytes || strings.TrimSpace(b.Key) != b.Key || strings.ContainsAny(b.Key, "\r\n") {
		return errors.New("artifact: invalid batch key")
	}
	if len(b.Artifacts)+len(b.Contents)+len(b.Manifests)+len(b.Lineage)+len(b.Aliases)+len(b.Locations) == 0 {
		return errors.New("artifact: empty batch")
	}
	for _, descriptor := range b.Artifacts {
		if err := descriptor.Validate(); err != nil {
			return err
		}
	}
	for _, content := range b.Contents {
		if err := content.Validate(); err != nil {
			return err
		}
	}
	for _, edge := range b.Lineage {
		if err := edge.Validate(); err != nil {
			return err
		}
	}
	for _, manifest := range b.Manifests {
		if err := manifest.Validate(); err != nil {
			return err
		}
	}
	for _, alias := range b.Aliases {
		if err := alias.Validate(); err != nil {
			return err
		}
	}
	for _, location := range b.Locations {
		if err := location.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// CommitID: hash-chained transaction identity
type CommitID [32]byte

func (id CommitID) String() string {
	return fmt.Sprintf("%x", id[:])
}

func (id CommitID) Valid() bool {
	return id != CommitID{}
}

// Reader: storage-neutral artifact queries
type Reader interface {
	Artifact(context.Context, ID) (Descriptor, bool, error)
	Content(context.Context, ID) (Content, bool, error)
	Manifest(context.Context, ID) (Manifest, bool, error)
	ResolveAlias(context.Context, string) (ID, bool, error)
	Parents(context.Context, ID) ([]Lineage, error)
	Children(context.Context, ID) ([]Lineage, error)
	Locations(context.Context, ID) ([]Location, error)
}

// Repository: transactional artifact catalog
type Repository interface {
	Reader
	Commit(context.Context, Batch) (CommitID, error)
	Close() error
}

func CommitBatch(ctx context.Context, repository Repository, batch Batch) (CommitID, error) {
	if ctx == nil || repository == nil {
		return CommitID{}, errors.New("artifact: nil commit context or repository")
	}
	if err := ctx.Err(); err != nil {
		return CommitID{}, err
	}
	if err := batch.Validate(); err != nil {
		return CommitID{}, err
	}
	return repository.Commit(ctx, batch)
}

func ResolveAlias(ctx context.Context, reader Reader, name string) (ID, bool, error) {
	if ctx == nil || reader == nil {
		return ID{}, false, errors.New("artifact: nil alias context or reader")
	}
	if err := ctx.Err(); err != nil {
		return ID{}, false, err
	}
	return reader.ResolveAlias(ctx, name)
}
