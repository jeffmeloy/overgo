package dataset

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/strictjson"
)

const (
	Version   uint16 = 1
	MediaType        = "application/vnd.llamacpp2go.dataset+json"
	Schema           = "llamacpp2go/dataset/v1"

	maxEntries   = 1 << 20
	maxNameBytes = 1024
)

// Type: canonical dataset document form.
type Type string

const (
	TypeVersion Type = "version"
	TypeView    Type = "view"
	TypeSplit   Type = "split"
	TypeMixture Type = "mixture"
)

// Asset: version-owned external data component.
type Asset struct {
	Name     string      `json:"name"`
	Artifact artifact.ID `json:"artifact"`
	Records  uint64      `json:"records"`
}

// Partition: named split view.
type Partition struct {
	Name string      `json:"name"`
	View artifact.ID `json:"view"`
}

// Member: normalized mixture contribution.
type Member struct {
	Dataset artifact.ID `json:"dataset"`
	Weight  uint64      `json:"weight"`
}

// Document: immutable dataset composition fact.
type Document struct {
	Version    uint16       `json:"version"`
	Type       Type         `json:"type"`
	Assets     []Asset      `json:"assets,omitempty"`
	Source     *artifact.ID `json:"source,omitempty"`
	Selector   *artifact.ID `json:"selector,omitempty"`
	Fields     []string     `json:"fields,omitempty"`
	Partitions []Partition  `json:"partitions,omitempty"`
	Members    []Member     `json:"members,omitempty"`
	ID         artifact.ID  `json:"-"`
}

func NewVersion(assets []Asset) (Document, error) {
	return newDocument(Document{Version: Version, Type: TypeVersion, Assets: slices.Clone(assets)})
}

func NewView(source artifact.ID, selector *artifact.ID, fields []string) (Document, error) {
	return newDocument(Document{
		Version: Version, Type: TypeView, Source: cloneID(idPointer(source)),
		Selector: cloneID(selector), Fields: slices.Clone(fields),
	})
}

func NewSplit(source artifact.ID, partitions []Partition) (Document, error) {
	return newDocument(Document{
		Version: Version, Type: TypeSplit, Source: cloneID(idPointer(source)),
		Partitions: slices.Clone(partitions),
	})
}

func NewMixture(members []Member) (Document, error) {
	return newDocument(Document{Version: Version, Type: TypeMixture, Members: slices.Clone(members)})
}

func Parse(content []byte) (Document, error) {
	var document Document
	if err := strictjson.DecodeBytes(content, &document); err != nil {
		return Document{}, fmt.Errorf("dataset: decode document: %w", err)
	}
	parsed, err := newDocument(document)
	if err != nil {
		return Document{}, err
	}
	canonical, err := parsed.ContentBytes()
	if err != nil {
		return Document{}, err
	}
	if !bytes.Equal(canonical, content) {
		return Document{}, errors.New("dataset: non-canonical document")
	}
	return parsed, nil
}

func (d Document) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindDataset {
		return errors.New("dataset: invalid document identity")
	}
	canonical := d
	canonical.ID = artifact.ID{}
	if err := canonicalize(&canonical); err != nil {
		return err
	}
	canonical.ID = d.ID
	if !sameDocument(d, canonical) {
		return errors.New("dataset: document is not canonical")
	}
	canonical.ID = artifact.ID{}
	content, err := json.Marshal(canonical)
	if err != nil {
		return fmt.Errorf("dataset: encode document: %w", err)
	}
	want, err := artifact.IdentifyBytes(artifact.KindDataset, content)
	if err != nil {
		return err
	}
	if d.ID != want {
		return errors.New("dataset: document identity mismatch")
	}
	return nil
}

func (d Document) ContentBytes() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	canonical := d
	canonical.ID = artifact.ID{}
	content, err := json.Marshal(canonical)
	if err != nil {
		return nil, fmt.Errorf("dataset: encode document: %w", err)
	}
	return content, nil
}

func (d Document) Content() (artifact.Content, error) {
	content, err := d.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return artifact.Content{Descriptor: artifact.Descriptor{
		ID: d.ID, Size: uint64(len(content)), MediaType: MediaType, Schema: Schema,
	}, Data: content}, nil
}

func (d Document) Lineage() []artifact.Lineage {
	edges := make([]artifact.Lineage, 0, len(d.Assets)+len(d.Partitions)+len(d.Members)+2)
	appendEdge := func(parent artifact.ID, relation artifact.Relation) {
		edges = append(edges, artifact.Lineage{Child: d.ID, Parent: parent, Relation: relation})
	}
	for _, asset := range d.Assets {
		appendEdge(asset.Artifact, artifact.RelationContains)
	}
	if d.Source != nil {
		appendEdge(*d.Source, artifact.RelationDerivedFrom)
	}
	if d.Selector != nil {
		appendEdge(*d.Selector, artifact.RelationDependsOn)
	}
	for _, partition := range d.Partitions {
		appendEdge(partition.View, artifact.RelationContains)
	}
	for _, member := range d.Members {
		appendEdge(member.Dataset, artifact.RelationDependsOn)
	}
	return edges
}

func newDocument(document Document) (Document, error) {
	document.ID = artifact.ID{}
	if err := canonicalize(&document); err != nil {
		return Document{}, err
	}
	content, err := json.Marshal(document)
	if err != nil {
		return Document{}, fmt.Errorf("dataset: encode document: %w", err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindDataset, content)
	if err != nil {
		return Document{}, err
	}
	document.ID = id
	return document, nil
}

func canonicalize(document *Document) error {
	if document == nil || document.Version != Version {
		return errors.New("dataset: invalid document version")
	}
	document.Source = cloneID(document.Source)
	document.Selector = cloneID(document.Selector)
	switch document.Type {
	case TypeVersion:
		if document.Source != nil || document.Selector != nil || len(document.Fields)+len(document.Partitions)+len(document.Members) != 0 {
			return errors.New("dataset: version has incompatible fields")
		}
		return canonicalizeAssets(&document.Assets)
	case TypeView:
		if len(document.Assets)+len(document.Partitions)+len(document.Members) != 0 || !datasetID(document.Source) ||
			document.Selector == nil && len(document.Fields) == 0 {
			return errors.New("dataset: invalid view")
		}
		if document.Selector != nil && !document.Selector.Valid() {
			return errors.New("dataset: invalid view selector")
		}
		return canonicalizeNames(&document.Fields)
	case TypeSplit:
		if len(document.Assets)+len(document.Fields)+len(document.Members) != 0 || document.Selector != nil || !datasetID(document.Source) {
			return errors.New("dataset: invalid split")
		}
		return canonicalizePartitions(&document.Partitions)
	case TypeMixture:
		if len(document.Assets)+len(document.Fields)+len(document.Partitions) != 0 || document.Source != nil || document.Selector != nil {
			return errors.New("dataset: mixture has incompatible fields")
		}
		return canonicalizeMembers(&document.Members)
	default:
		return errors.New("dataset: invalid document type")
	}
}

func canonicalizeAssets(assets *[]Asset) error {
	if len(*assets) == 0 || len(*assets) > maxEntries {
		return errors.New("dataset: invalid asset count")
	}
	sort.Slice(*assets, func(i, j int) bool { return (*assets)[i].Name < (*assets)[j].Name })
	for index, asset := range *assets {
		if !validName(asset.Name) || !asset.Artifact.Valid() ||
			asset.Artifact.Kind() != artifact.KindDatasetShard && asset.Artifact.Kind() != artifact.KindFile {
			return errors.New("dataset: invalid asset")
		}
		if index > 0 && (*assets)[index-1].Name == asset.Name {
			return fmt.Errorf("dataset: duplicate asset %q", asset.Name)
		}
	}
	return nil
}

func canonicalizeNames(names *[]string) error {
	if len(*names) > maxEntries {
		return errors.New("dataset: too many fields")
	}
	slices.Sort(*names)
	for index, name := range *names {
		if !validName(name) {
			return errors.New("dataset: invalid field")
		}
		if index > 0 && (*names)[index-1] == name {
			return fmt.Errorf("dataset: duplicate field %q", name)
		}
	}
	return nil
}

func canonicalizePartitions(partitions *[]Partition) error {
	if len(*partitions) < 2 || len(*partitions) > maxEntries {
		return errors.New("dataset: invalid partition count")
	}
	sort.Slice(*partitions, func(i, j int) bool { return (*partitions)[i].Name < (*partitions)[j].Name })
	seen := make(map[artifact.ID]struct{}, len(*partitions))
	for index, partition := range *partitions {
		if !validName(partition.Name) || partition.View.Kind() != artifact.KindDataset {
			return errors.New("dataset: invalid partition")
		}
		if index > 0 && (*partitions)[index-1].Name == partition.Name {
			return fmt.Errorf("dataset: duplicate partition %q", partition.Name)
		}
		if _, duplicate := seen[partition.View]; duplicate {
			return fmt.Errorf("dataset: duplicate partition view %s", partition.View)
		}
		seen[partition.View] = struct{}{}
	}
	return nil
}

func canonicalizeMembers(members *[]Member) error {
	if len(*members) < 2 || len(*members) > maxEntries {
		return errors.New("dataset: invalid mixture member count")
	}
	sort.Slice(*members, func(i, j int) bool {
		return (*members)[i].Dataset.String() < (*members)[j].Dataset.String()
	})
	divisor := uint64(0)
	for index, member := range *members {
		if member.Dataset.Kind() != artifact.KindDataset || member.Weight == 0 {
			return errors.New("dataset: invalid mixture member")
		}
		if index > 0 && (*members)[index-1].Dataset == member.Dataset {
			return fmt.Errorf("dataset: duplicate mixture member %s", member.Dataset)
		}
		divisor = gcd(divisor, member.Weight)
	}
	for index := range *members {
		(*members)[index].Weight /= divisor
	}
	return nil
}

func sameDocument(left, right Document) bool {
	return left.Version == right.Version && left.Type == right.Type && left.ID == right.ID &&
		sameID(left.Source, right.Source) && sameID(left.Selector, right.Selector) &&
		slices.Equal(left.Assets, right.Assets) && slices.Equal(left.Fields, right.Fields) &&
		slices.Equal(left.Partitions, right.Partitions) && slices.Equal(left.Members, right.Members)
}

func datasetID(id *artifact.ID) bool {
	return id != nil && id.Kind() == artifact.KindDataset
}

func idPointer(id artifact.ID) *artifact.ID { return &id }

func cloneID(id *artifact.ID) *artifact.ID {
	if id == nil {
		return nil
	}
	copy := *id
	return &copy
}

func sameID(left, right *artifact.ID) bool {
	return left == nil && right == nil || left != nil && right != nil && *left == *right
}

func validName(value string) bool {
	return value != "" && len(value) <= maxNameBytes && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\r\n\\")
}

func gcd(left, right uint64) uint64 {
	for right != 0 {
		left, right = right, left%right
	}
	return left
}
