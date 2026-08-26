package modelartifact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
	tensorgraph "overgo/internal/tensor"
)

const (
	TensorInventoryVersion   uint16 = 2
	TensorInventoryMediaType        = "application/vnd.overgo.tensor-inventory+json"
	TensorInventorySchema           = "overgo/tensor-inventory/v2"
)

var tensorInventoryContract = artifact.DocumentContract{
	Kind: artifact.KindTensorInventory, MediaType: TensorInventoryMediaType, Schema: TensorInventorySchema,
}

var tensorInventoryCodec = artifact.DocumentCodec[TensorInventoryDocument]{
	Name: "tensor inventory", Contract: tensorInventoryContract,
	Decode: func(data []byte, value *TensorInventoryDocument) error {
		var body tensorInventoryBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = TensorInventoryDocument{
			Version: body.Version, Owner: body.Owner, Format: body.Format, Tensors: body.Tensors,
		}
		return nil
	},
	Encode: tensorInventoryContent,
	Canonicalize: func(value *TensorInventoryDocument) error {
		return value.validate()
	},
	Clone: func(value TensorInventoryDocument) TensorInventoryDocument {
		value.Tensors = cloneTensorFacts(value.Tensors)
		return value
	},
	Identity:    func(value TensorInventoryDocument) artifact.ID { return value.ID },
	SetIdentity: func(value *TensorInventoryDocument, id artifact.ID) { value.ID = id },
}

// TensorFormat: source representation contract.
type TensorFormat string

const (
	TensorFormatGGUF        TensorFormat = "gguf"
	TensorFormatSafetensors TensorFormat = "safetensors"
	TensorFormatPyTorch     TensorFormat = "pytorch-zip"
	TensorFormatMixed       TensorFormat = "mixed"
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
	Owner   artifact.ID  `json:"owner"`
	Format  TensorFormat `json:"format"`
	Tensors []TensorFact `json:"tensors"`
}

// TensorInventoryDocument: immutable artifact tensor facts.
type TensorInventoryDocument struct {
	ID      artifact.ID
	Version uint16
	Owner   artifact.ID
	Format  TensorFormat
	Tensors []TensorFact
}

func NewTensorInventoryDocument(
	owner artifact.ID,
	format TensorFormat,
	tensors []TensorFact,
) (TensorInventoryDocument, error) {
	document := TensorInventoryDocument{
		Version: TensorInventoryVersion,
		Owner:   owner,
		Format:  format,
		Tensors: cloneTensorFacts(tensors),
	}
	return tensorInventoryCodec.New(document)
}

func ReadTensorInventoryDocument(ctx context.Context, store artifact.Reader, id artifact.ID) (TensorInventoryDocument, bool, error) {
	return tensorInventoryCodec.Read(ctx, store, id)
}

func (d TensorInventoryDocument) ValidateIdentity() error {
	return tensorInventoryCodec.ValidateIdentity(d)
}

func (d TensorInventoryDocument) Content() (artifact.Content, error) {
	return tensorInventoryCodec.Content(d)
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

// LoadTensorInventory resolves the unique inventory derived from its owner.
func LoadTensorInventory(
	ctx context.Context,
	store artifact.Reader,
	owner artifact.ID,
) (TensorInventoryDocument, bool, error) {
	if store == nil || !tensorBearingKind(owner.Kind()) {
		return TensorInventoryDocument{}, false, errors.New("tensor inventory: invalid owner lookup")
	}
	edges, err := store.Children(ctx, owner)
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
			return TensorInventoryDocument{}, false, errors.New("tensor inventory: multiple owner inventories")
		}
		document, ok, contentErr := tensorInventoryCodec.Read(ctx, store, edge.Child)
		if contentErr != nil {
			return TensorInventoryDocument{}, false, contentErr
		}
		if !ok {
			return TensorInventoryDocument{}, false, errors.New("tensor inventory: content is absent or incompatible")
		}
		found = document
		if found.Owner != owner || found.ID != edge.Child {
			return TensorInventoryDocument{}, false, errors.New("tensor inventory: owner lineage mismatch")
		}
		matched = true
	}
	return found, matched, nil
}

func (d TensorInventoryDocument) validate() error {
	if d.Version != TensorInventoryVersion || !tensorBearingKind(d.Owner.Kind()) {
		return errors.New("tensor inventory: invalid envelope")
	}
	if d.Format != TensorFormatGGUF && d.Format != TensorFormatSafetensors &&
		d.Format != TensorFormatPyTorch && d.Format != TensorFormatMixed {
		return errors.New("tensor inventory: invalid format")
	}
	if len(d.Tensors) == 0 {
		return errors.New("tensor inventory: empty")
	}
	previous := ""
	for _, tensor := range d.Tensors {
		if tensor.Name <= previous ||
			strings.TrimSpace(tensor.Name) != tensor.Name || strings.ContainsAny(tensor.Name, "\r\n") {
			return errors.New("tensor inventory: invalid or unordered tensor name")
		}
		if tensor.Storage == "" ||
			tensor.Storage != strings.ToLower(tensor.Storage) || strings.TrimSpace(tensor.Storage) != tensor.Storage {
			return fmt.Errorf("tensor inventory: tensor %q has invalid storage", tensor.Name)
		}
		if tensor.Shape == nil || !checked.Nonzero(tensor.Bytes) {
			return fmt.Errorf("tensor inventory: tensor %q has invalid shape or byte extent", tensor.Name)
		}
		elements := uint64(tensorgraph.SingletonExtent)
		for _, dimension := range tensor.Shape {
			var ok bool
			elements, ok = checked.Mul64(elements, dimension)
			if !ok || !checked.Nonzero(elements) {
				return fmt.Errorf("tensor inventory: tensor %q shape extent is invalid", tensor.Name)
			}
		}
		previous = tensor.Name
	}
	return nil
}

func tensorInventoryContent(document TensorInventoryDocument) ([]byte, error) {
	content, err := json.Marshal(tensorInventoryBody{
		Version: document.Version, Owner: document.Owner, Format: document.Format,
		Tensors: document.Tensors,
	})
	if err != nil {
		return nil, fmt.Errorf("tensor inventory: encode: %w", err)
	}
	return content, nil
}

// NewGGUFTensorInventory captures logical facts from a validated GGUF.
func NewGGUFTensorInventory(owner artifact.ID, file *gguf.File) (TensorInventoryDocument, error) {
	if file == nil {
		return TensorInventoryDocument{}, errors.New("tensor inventory: nil GGUF source")
	}
	facts := make([]TensorFact, len(file.Tensors))
	for index, tensor := range file.Tensors {
		facts[index] = TensorFact{
			Name: tensor.Name, Shape: slices.Clone(tensor.Shape[:tensor.Dimensions]),
			Storage: strings.ToLower(tensor.Type.String()), Bytes: tensor.Size,
		}
	}
	sort.Slice(facts, func(left, right int) bool { return facts[left].Name < facts[right].Name })
	return NewTensorInventoryDocument(owner, TensorFormatGGUF, facts)
}

// NewSafetensorsTensorInventory captures logical facts from a validated source.
func NewSafetensorsTensorInventory(owner artifact.ID, source *safetensors.Source) (TensorInventoryDocument, error) {
	if source == nil {
		return TensorInventoryDocument{}, errors.New("tensor inventory: nil Safetensors source")
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
	return NewTensorInventoryDocument(owner, TensorFormatSafetensors, facts)
}

func cloneTensorFacts(facts []TensorFact) []TensorFact {
	cloned := make([]TensorFact, len(facts))
	for index, fact := range facts {
		fact.Shape = slices.Clone(fact.Shape)
		cloned[index] = fact
	}
	return cloned
}
