package dataset

import (
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	MediaType = "application/vnd.overgo.dataset+json"
	Schema    = "overgo/dataset/v1"
)

var documentContract = artifact.DocumentContract{
	Kind: artifact.KindDataset, MediaType: MediaType, Schema: Schema,
}

var documentCodec = artifact.JSONDocumentCodec(
	"dataset", documentContract.Kind, documentContract.MediaType, documentContract.Schema,
	canonicalize, func(value Document) artifact.ID { return value.ID },
	func(value *Document, id artifact.ID) { value.ID = id }, cloneDocument,
)

// Type defines canonical dataset document form.
type Type string

const (
	TypeVersion Type = "version"
	TypeView    Type = "view"
	TypeSplit   Type = "split"
	TypeMixture Type = "mixture"
	// TypeCatalog indexes registered dataset versions.
	TypeCatalog Type = "catalog"
)

// Asset defines version-owned external data component.
type Asset struct {
	Name     string      `json:"name"`
	Artifact artifact.ID `json:"artifact"`
	Records  uint64      `json:"records"`
}

// Partition defines named split view.
type Partition struct {
	Name string      `json:"name"`
	View artifact.ID `json:"view"`
}

// Member defines normalized mixture contribution.
type Member struct {
	Dataset artifact.ID `json:"dataset"`
	Weight  uint64      `json:"weight"`
}

// CatalogEntry binds a stable name and inventory summary to one dataset.
type CatalogEntry struct {
	Name        string      `json:"name"`
	Dataset     artifact.ID `json:"dataset"`
	Inventory   artifact.ID `json:"inventory"`
	StorageKind string      `json:"storage_kind"`
	Source      string      `json:"source"`
	// Modality is the dataset's primary declared class: an explicit
	// declaration when one is given, "mixed" when the corpus spans more
	// than one modality, or the single modality present. It is never a
	// byte-volume guess -- Modalities carries the honest set.
	Modality string `json:"modality"`
	// Modalities is the set of distinct modalities the corpus contains,
	// so a mixed dataset is described faithfully rather than reduced to
	// one winner.
	Modalities []string `json:"modalities,omitempty"`
	Formats    []string `json:"formats,omitempty"`
	Files      uint64   `json:"files"`
	Bytes      uint64   `json:"bytes"`
}

// MixedModality is the primary class of a corpus that spans more than
// one modality with no explicit declaration.
const MixedModality = "mixed"

// Document defines immutable dataset composition fact.
type Document struct {
	Version    uint16         `json:"version"`
	Type       Type           `json:"type"`
	Assets     []Asset        `json:"assets,omitempty"`
	Source     *artifact.ID   `json:"source,omitempty"`
	Selector   *artifact.ID   `json:"selector,omitempty"`
	Fields     []string       `json:"fields,omitempty"`
	Partitions []Partition    `json:"partitions,omitempty"`
	Members    []Member       `json:"members,omitempty"`
	Catalog    []CatalogEntry `json:"catalog,omitempty"`
	ID         artifact.ID    `json:"-"`
}

func NewVersion(assets []Asset) (Document, error) {
	return newDocument(Document{Version: artifact.InitialDocumentVersion, Type: TypeVersion, Assets: slices.Clone(assets)})
}

func NewView(source artifact.ID, selector *artifact.ID, fields []string) (Document, error) {
	return newDocument(Document{
		Version: artifact.InitialDocumentVersion, Type: TypeView, Source: artifact.IDPointer(source),
		Selector: artifact.CloneID(selector), Fields: slices.Clone(fields),
	})
}

func NewSplit(source artifact.ID, partitions []Partition) (Document, error) {
	return newDocument(Document{
		Version: artifact.InitialDocumentVersion, Type: TypeSplit, Source: artifact.IDPointer(source),
		Partitions: slices.Clone(partitions),
	})
}

// NewMixture returns a normalized weighted dataset composition.
func NewMixture(members []Member) (Document, error) {
	return newDocument(Document{Version: artifact.InitialDocumentVersion, Type: TypeMixture, Members: slices.Clone(members)})
}

// NewCatalog returns a named dataset inventory index.
func NewCatalog(entries []CatalogEntry) (Document, error) {
	return newDocument(Document{Version: artifact.InitialDocumentVersion, Type: TypeCatalog, Catalog: cloneCatalog(entries)})
}

// Parse decodes and validates a dataset document.
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
	edges := make([]artifact.Lineage, 0, len(d.Assets)+len(d.Partitions)+len(d.Members)+len(d.Catalog)+2)
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
	for _, entry := range d.Catalog {
		appendEdge(entry.Dataset, artifact.RelationContains)
	}
	return edges
}

func newDocument(document Document) (Document, error) {
	return documentCodec.New(document)
}

func canonicalize(document *Document) error {
	if document == nil || document.Version != artifact.InitialDocumentVersion {
		return errors.New("dataset: invalid document version")
	}
	document.Source = artifact.CloneID(document.Source)
	document.Selector = artifact.CloneID(document.Selector)
	switch document.Type {
	case TypeVersion:
		if document.Source != nil || document.Selector != nil || len(document.Fields)+len(document.Partitions)+len(document.Members)+len(document.Catalog) != 0 {
			return errors.New("dataset: version has incompatible fields")
		}
		return canonicalizeAssets(&document.Assets)
	case TypeView:
		if len(document.Assets)+len(document.Partitions)+len(document.Members)+len(document.Catalog) != 0 || !datasetID(document.Source) ||
			document.Selector == nil && len(document.Fields) == 0 {
			return errors.New("dataset: invalid view")
		}
		if document.Selector != nil && !document.Selector.Valid() {
			return errors.New("dataset: invalid view selector")
		}
		return canonicalizeNames(&document.Fields)
	case TypeSplit:
		if len(document.Assets)+len(document.Fields)+len(document.Members)+len(document.Catalog) != 0 || document.Selector != nil || !datasetID(document.Source) {
			return errors.New("dataset: invalid split")
		}
		return canonicalizePartitions(&document.Partitions)
	case TypeMixture:
		if len(document.Assets)+len(document.Fields)+len(document.Partitions)+len(document.Catalog) != 0 || document.Source != nil || document.Selector != nil {
			return errors.New("dataset: mixture has incompatible fields")
		}
		return canonicalizeMembers(&document.Members)
	case TypeCatalog:
		if len(document.Assets)+len(document.Fields)+len(document.Partitions)+len(document.Members) != 0 || document.Source != nil || document.Selector != nil {
			return errors.New("dataset: catalog has incompatible fields")
		}
		return canonicalizeCatalog(&document.Catalog)
	default:
		return errors.New("dataset: invalid document type")
	}
}

func canonicalizeCatalog(entries *[]CatalogEntry) error {
	if len(*entries) == 0 {
		return errors.New("dataset: invalid catalog entry count")
	}
	sort.Slice(*entries, func(i, j int) bool { return (*entries)[i].Name < (*entries)[j].Name })
	for index := range *entries {
		entry := &(*entries)[index]
		slices.Sort(entry.Formats)
		entry.Formats = slices.Compact(entry.Formats)
		slices.Sort(entry.Modalities)
		entry.Modalities = slices.Compact(entry.Modalities)
		if !validName(entry.Name) || entry.Dataset.Kind() != artifact.KindDataset || entry.Inventory.Kind() != artifact.KindDatasetShard ||
			!validName(entry.StorageKind) || !validName(entry.Source) || !validName(entry.Modality) ||
			index > 0 && (*entries)[index-1].Name == entry.Name {
			return errors.New("dataset: invalid catalog entry")
		}
		for _, modality := range entry.Modalities {
			if !validName(modality) {
				return errors.New("dataset: invalid catalog entry modality")
			}
		}
		for _, format := range entry.Formats {
			if !validName(format) {
				return errors.New("dataset: invalid catalog format")
			}
		}
	}
	return nil
}

func canonicalizeAssets(assets *[]Asset) error {
	if len(*assets) == 0 {
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
	if len(*partitions) < 2 {
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
	if len(*members) < 2 {
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

func datasetID(id *artifact.ID) bool {
	return id != nil && id.Kind() == artifact.KindDataset
}

func cloneDocument(document Document) Document {
	document.Assets = slices.Clone(document.Assets)
	document.Source = artifact.CloneID(document.Source)
	document.Selector = artifact.CloneID(document.Selector)
	document.Fields = slices.Clone(document.Fields)
	document.Partitions = slices.Clone(document.Partitions)
	document.Members = slices.Clone(document.Members)
	document.Catalog = cloneCatalog(document.Catalog)
	return document
}

func cloneCatalog(entries []CatalogEntry) []CatalogEntry {
	result := slices.Clone(entries)
	for index := range result {
		result[index].Formats = slices.Clone(result[index].Formats)
	}
	return result
}

func validName(value string) bool {
	return textcheck.Bounded(value, artifact.MaxContentBytes, "\r\n\\")
}

func gcd(left, right uint64) uint64 {
	for right != 0 {
		left, right = right, left%right
	}
	return left
}
