package modelartifact

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
)

const (
	TensorInventoryVersion   uint16 = 1
	TensorInventoryMediaType        = "application/vnd.overgo.tensor-inventory+json"
	TensorInventorySchema           = "overgo/tensor-inventory/v1"
	maxInventoryTensors             = 500_000
	maxInventoryRank                = 64
	maxInventoryNameBytes           = 16 << 10
	maxInventoryStorageBytes        = 64
)

var tensorInventoryContract = artifact.DocumentContract{
	Kind: artifact.KindTensorInventory, MediaType: TensorInventoryMediaType, Schema: TensorInventorySchema,
}

func TensorInventoryDocumentContract() artifact.DocumentContract { return tensorInventoryContract }

// TensorFormat: source representation contract.
type TensorFormat string

const (
	TensorFormatGGUF        TensorFormat = "gguf"
	TensorFormatSafetensors TensorFormat = "safetensors"
)

// TensorFact: logical tensor storage and shape.
type TensorFact struct {
	Name    string   `json:"name"`
	Shape   []uint64 `json:"shape"`
	Storage string   `json:"storage"`
	Bytes   uint64   `json:"bytes"`
}

type tensorInventoryBody struct {
	Version uint16       `json:"version"`
	Model   artifact.ID  `json:"model"`
	Format  TensorFormat `json:"format"`
	Tensors []TensorFact `json:"tensors"`
}

// TensorInventoryDocument: immutable model tensor facts.
type TensorInventoryDocument struct {
	ID      artifact.ID
	Version uint16
	Model   artifact.ID
	Format  TensorFormat
	Tensors []TensorFact
}

func NewTensorInventoryDocument(
	model artifact.ID,
	format TensorFormat,
	tensors []TensorFact,
) (TensorInventoryDocument, error) {
	document := TensorInventoryDocument{
		Version: TensorInventoryVersion,
		Model:   model,
		Format:  format,
		Tensors: cloneTensorFacts(tensors),
	}
	if err := document.validateShape(); err != nil {
		return TensorInventoryDocument{}, err
	}
	content, err := tensorInventoryContent(document)
	if err != nil {
		return TensorInventoryDocument{}, err
	}
	if len(content) > artifact.MaxContentBytes {
		return TensorInventoryDocument{}, errors.New("model artifact: tensor inventory exceeds inline content limit")
	}
	document.ID, err = tensorInventoryContract.Identify(content)
	return document, err
}

func ParseTensorInventoryDocument(content []byte) (TensorInventoryDocument, error) {
	var body tensorInventoryBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return TensorInventoryDocument{}, fmt.Errorf("model artifact: decode tensor inventory: %w", err)
	}
	document, err := NewTensorInventoryDocument(body.Model, body.Format, body.Tensors)
	if err != nil {
		return TensorInventoryDocument{}, err
	}
	if body.Version != TensorInventoryVersion {
		return TensorInventoryDocument{}, errors.New("model artifact: unsupported tensor inventory version")
	}
	canonical, err := document.ContentBytes()
	if err != nil {
		return TensorInventoryDocument{}, err
	}
	if !bytes.Equal(content, canonical) {
		return TensorInventoryDocument{}, errors.New("model artifact: non-canonical tensor inventory")
	}
	return document, nil
}

func (d TensorInventoryDocument) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindTensorInventory {
		return errors.New("model artifact: invalid tensor inventory identity")
	}
	if err := d.validateShape(); err != nil {
		return err
	}
	content, err := tensorInventoryContent(d)
	if err != nil {
		return err
	}
	if err := tensorInventoryContract.ValidateIdentity(d.ID, content); err != nil {
		return errors.New("model artifact: tensor inventory identity mismatch")
	}
	return nil
}

func (d TensorInventoryDocument) ContentBytes() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	return tensorInventoryContent(d)
}

func (d TensorInventoryDocument) Content() (artifact.Content, error) {
	content, err := d.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return tensorInventoryContract.Content(d.ID, content)
}

func (d TensorInventoryDocument) Tensor(name string) (TensorFact, bool) {
	index, found := slices.BinarySearchFunc(d.Tensors, name, func(fact TensorFact, target string) int {
		return strings.Compare(fact.Name, target)
	})
	if !found {
		return TensorFact{}, false
	}
	fact := d.Tensors[index]
	fact.Shape = slices.Clone(fact.Shape)
	return fact, true
}

// LoadTensorInventory resolves the unique inventory derived from a model.
func LoadTensorInventory(
	ctx context.Context,
	store artifact.Reader,
	model artifact.ID,
) (TensorInventoryDocument, bool, error) {
	if store == nil || model.Kind() != artifact.KindModel {
		return TensorInventoryDocument{}, false, errors.New("model artifact: invalid tensor inventory lookup")
	}
	edges, err := store.Children(ctx, model)
	if err != nil {
		return TensorInventoryDocument{}, false, err
	}
	var found TensorInventoryDocument
	matched := false
	for _, edge := range edges {
		if edge.Relation != artifact.RelationDerivedFrom || edge.Child.Kind() != artifact.KindTensorInventory {
			continue
		}
		if matched {
			return TensorInventoryDocument{}, false, errors.New("model artifact: multiple tensor inventories")
		}
		content, ok, contentErr := artifact.ReadDocument(ctx, store, edge.Child, tensorInventoryContract)
		if contentErr != nil {
			return TensorInventoryDocument{}, false, contentErr
		}
		if !ok {
			return TensorInventoryDocument{}, false, errors.New("model artifact: tensor inventory content is absent or incompatible")
		}
		found, err = ParseTensorInventoryDocument(content.Data)
		if err != nil {
			return TensorInventoryDocument{}, false, err
		}
		if found.Model != model || found.ID != edge.Child {
			return TensorInventoryDocument{}, false, errors.New("model artifact: tensor inventory lineage mismatch")
		}
		matched = true
	}
	return found, matched, nil
}

func (d TensorInventoryDocument) validateShape() error {
	if d.Version != TensorInventoryVersion || d.Model.Kind() != artifact.KindModel {
		return errors.New("model artifact: invalid tensor inventory envelope")
	}
	if d.Format != TensorFormatGGUF && d.Format != TensorFormatSafetensors {
		return errors.New("model artifact: invalid tensor inventory format")
	}
	if len(d.Tensors) == 0 || len(d.Tensors) > maxInventoryTensors {
		return errors.New("model artifact: invalid tensor inventory count")
	}
	previous := ""
	for _, tensor := range d.Tensors {
		if tensor.Name <= previous || len(tensor.Name) > maxInventoryNameBytes ||
			strings.TrimSpace(tensor.Name) != tensor.Name || strings.ContainsAny(tensor.Name, "\r\n") {
			return errors.New("model artifact: invalid or unordered tensor name")
		}
		if tensor.Storage == "" || len(tensor.Storage) > maxInventoryStorageBytes ||
			tensor.Storage != strings.ToLower(tensor.Storage) || strings.TrimSpace(tensor.Storage) != tensor.Storage {
			return fmt.Errorf("model artifact: tensor %q has invalid storage", tensor.Name)
		}
		if tensor.Shape == nil || len(tensor.Shape) > maxInventoryRank {
			return fmt.Errorf("model artifact: tensor %q exceeds rank limit", tensor.Name)
		}
		previous = tensor.Name
	}
	return nil
}

func tensorInventoryContent(document TensorInventoryDocument) ([]byte, error) {
	content, err := json.Marshal(tensorInventoryBody{
		Version: document.Version, Model: document.Model, Format: document.Format,
		Tensors: document.Tensors,
	})
	if err != nil {
		return nil, fmt.Errorf("model artifact: encode tensor inventory: %w", err)
	}
	return content, nil
}

// NewGGUFTensorInventory captures logical facts from a validated GGUF.
func NewGGUFTensorInventory(model artifact.ID, file *gguf.File) (TensorInventoryDocument, error) {
	if file == nil {
		return TensorInventoryDocument{}, errors.New("model artifact: nil GGUF tensor inventory source")
	}
	facts := make([]TensorFact, len(file.Tensors))
	for index, tensor := range file.Tensors {
		facts[index] = TensorFact{
			Name: tensor.Name, Shape: slices.Clone(tensor.Shape[:tensor.Dimensions]),
			Storage: strings.ToLower(tensor.Type.String()), Bytes: tensor.Size,
		}
	}
	sort.Slice(facts, func(left, right int) bool { return facts[left].Name < facts[right].Name })
	return NewTensorInventoryDocument(model, TensorFormatGGUF, facts)
}

// NewSafetensorsTensorInventory captures logical facts from a validated source.
func NewSafetensorsTensorInventory(model artifact.ID, source *safetensors.Source) (TensorInventoryDocument, error) {
	if source == nil {
		return TensorInventoryDocument{}, errors.New("model artifact: nil Safetensors tensor inventory source")
	}
	names := source.Names()
	facts := make([]TensorFact, len(names))
	for index, name := range names {
		tensor := source.Tensors[name]
		facts[index] = TensorFact{
			Name: name, Shape: slices.Clone(tensor.Shape),
			Storage: strings.ToLower(tensor.DType), Bytes: uint64(tensor.Size()),
		}
	}
	return NewTensorInventoryDocument(model, TensorFormatSafetensors, facts)
}

func cloneTensorFacts(facts []TensorFact) []TensorFact {
	cloned := make([]TensorFact, len(facts))
	for index, fact := range facts {
		fact.Shape = slices.Clone(fact.Shape)
		cloned[index] = fact
	}
	return cloned
}
