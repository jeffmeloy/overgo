package artifact

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"overgo/internal/strictjson"
)

// DocumentCodec defines canonical typed lifecycle.
type DocumentCodec[T any] struct {
	Name         string
	Contract     DocumentContract
	ContractFor  func(T) DocumentContract
	Decode       func([]byte, *T) error
	Encode       func(T) ([]byte, error)
	Canonicalize func(*T) error
	Clone        func(T) T
	Identity     func(T) ID
	SetIdentity  func(*T, ID)
}

// JSONDocumentCodec returns a strict JSON codec with caller-owned schema and cloning.
func JSONDocumentCodec[T any](
	name string,
	kind Kind,
	mediaType, schema string,
	canonicalize func(*T) error,
	identity func(T) ID,
	setIdentity func(*T, ID),
	clone func(T) T,
) DocumentCodec[T] {
	return DocumentCodec[T]{
		Name: name,
		Contract: DocumentContract{
			Kind: kind, MediaType: mediaType, Schema: schema,
		},
		Decode:       func(data []byte, value *T) error { return strictjson.DecodeBytes(data, value) },
		Encode:       func(value T) ([]byte, error) { return json.Marshal(value) },
		Canonicalize: canonicalize,
		Clone:        clone,
		Identity:     identity,
		SetIdentity:  setIdentity,
	}
}

func (c DocumentCodec[T]) New(value T) (T, error) {
	if err := c.validate(); err != nil {
		return value, err
	}
	value, data, contract, err := c.canonical(value)
	if err != nil {
		return value, err
	}
	id, err := contract.Identify(data)
	if err != nil {
		return value, err
	}
	c.SetIdentity(&value, id)
	return value, nil
}

func (c DocumentCodec[T]) Parse(data []byte) (T, error) {
	if err := c.validate(); err != nil {
		var zero T
		return zero, err
	}
	value, canonical, contract, err := c.parseCanonical(data)
	if err != nil {
		return value, err
	}
	id, err := contract.Identify(canonical)
	if err != nil {
		return value, err
	}
	c.SetIdentity(&value, id)
	return value, nil
}

// Read returns a validated typed document without an intermediate content clone.
func (c DocumentCodec[T]) Read(ctx context.Context, reader Reader, id ID) (T, bool, error) {
	var zero T
	if err := c.validate(); err != nil {
		return zero, false, err
	}
	if ctx == nil || reader == nil {
		return zero, false, errors.New("artifact: nil document reader or context")
	}
	if err := ctx.Err(); err != nil {
		return zero, false, err
	}
	content, ok, err := reader.Content(ctx, id)
	if err != nil || !ok {
		return zero, ok, err
	}
	if err := content.Validate(); err != nil {
		return zero, false, fmt.Errorf("artifact: read document: %w", err)
	}
	value, _, contract, err := c.parseCanonical(content.Data)
	if err != nil {
		return zero, false, err
	}
	if err := contract.validateDescriptor(content.Descriptor, id); err != nil {
		return zero, false, fmt.Errorf("artifact: read document: %w", err)
	}
	c.SetIdentity(&value, id)
	return value, true, nil
}

// Require returns a validated typed document and rejects absence.
func (c DocumentCodec[T]) Require(ctx context.Context, reader Reader, id ID) (T, error) {
	value, ok, err := c.Read(ctx, reader, id)
	if err != nil {
		return value, err
	}
	if !ok {
		return value, fmt.Errorf("%s: document is absent", c.Name)
	}
	return value, nil
}

// Normalize converts external bytes to a canonical value and identity bytes.
func (c DocumentCodec[T]) Normalize(data []byte) (T, []byte, error) {
	var decoded T
	if err := c.validate(); err != nil {
		return decoded, nil, err
	}
	if err := c.Decode(data, &decoded); err != nil {
		return decoded, nil, fmt.Errorf("%s: decode document: %w", c.Name, err)
	}
	value, canonical, contract, err := c.canonical(decoded)
	if err != nil {
		return decoded, nil, err
	}
	id, err := contract.Identify(canonical)
	if err != nil {
		return decoded, nil, err
	}
	c.SetIdentity(&value, id)
	return value, canonical, nil
}

func (c DocumentCodec[T]) ValidateIdentity(value T) error {
	_, _, err := c.validated(value)
	return err
}

func (c DocumentCodec[T]) validated(value T) ([]byte, DocumentContract, error) {
	if err := c.validate(); err != nil {
		return nil, DocumentContract{}, err
	}
	id := c.Identity(value)
	canonical, data, contract, err := c.canonical(value)
	if err != nil {
		return nil, DocumentContract{}, err
	}
	c.SetIdentity(&canonical, id)
	if id.Kind() != contract.Kind {
		return nil, DocumentContract{}, fmt.Errorf("%s: invalid document identity", c.Name)
	}
	if !reflect.DeepEqual(value, canonical) {
		return nil, DocumentContract{}, fmt.Errorf("%s: document is not canonical", c.Name)
	}
	if err := contract.ValidateIdentity(id, data); err != nil {
		return nil, DocumentContract{}, fmt.Errorf("%s: document identity mismatch", c.Name)
	}
	return data, contract, nil
}

func (c DocumentCodec[T]) ContentBytes(value T) ([]byte, error) {
	data, _, err := c.validated(value)
	return data, err
}

func (c DocumentCodec[T]) Content(value T) (Content, error) {
	id := c.Identity(value)
	data, contract, err := c.validated(value)
	if err != nil {
		return Content{}, err
	}
	return contract.Content(id, data)
}

// Batch returns one canonical document with its publication edges.
func (c DocumentCodec[T]) Batch(key string, value T, lineage []Lineage, aliases []AliasBinding) (Batch, error) {
	content, err := c.Content(value)
	if err != nil {
		return Batch{}, err
	}
	return NewDocumentBatch(key, []Content{content}, lineage, aliases)
}

func (c DocumentCodec[T]) canonical(value T) (T, []byte, DocumentContract, error) {
	value = c.clone(value)
	c.SetIdentity(&value, ID{})
	if err := c.Canonicalize(&value); err != nil {
		return value, nil, DocumentContract{}, err
	}
	data, err := c.Encode(value)
	if err != nil {
		return value, nil, DocumentContract{}, err
	}
	return value, data, c.contract(value), nil
}

func (c DocumentCodec[T]) parseCanonical(data []byte) (T, []byte, DocumentContract, error) {
	var decoded T
	if err := c.Decode(data, &decoded); err != nil {
		return decoded, nil, DocumentContract{}, fmt.Errorf("%s: decode document: %w", c.Name, err)
	}
	value, canonical, contract, err := c.canonical(decoded)
	if err != nil {
		return decoded, nil, DocumentContract{}, err
	}
	if !bytes.Equal(canonical, data) {
		return decoded, nil, DocumentContract{}, fmt.Errorf("%s: non-canonical document", c.Name)
	}
	return value, canonical, contract, nil
}

func (c DocumentCodec[T]) clone(value T) T {
	if c.Clone != nil {
		return c.Clone(value)
	}
	return value
}

func (c DocumentCodec[T]) validate() error {
	if c.Name == "" || c.Decode == nil || c.Encode == nil || c.Canonicalize == nil ||
		c.Identity == nil || c.SetIdentity == nil || c.Contract.Kind == KindInvalid && c.ContractFor == nil {
		return errors.New("artifact: incomplete document codec")
	}
	return nil
}

func (c DocumentCodec[T]) contract(value T) DocumentContract {
	if c.ContractFor != nil {
		return c.ContractFor(value)
	}
	return c.Contract
}
