package artifact

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
)

// DocumentCodec: canonical typed document lifecycle.
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
	var decoded T
	if err := c.validate(); err != nil {
		return decoded, err
	}
	if err := c.Decode(data, &decoded); err != nil {
		return decoded, fmt.Errorf("%s: decode document: %w", c.Name, err)
	}
	value, canonical, contract, err := c.canonical(decoded)
	if err != nil {
		return decoded, err
	}
	if !bytes.Equal(canonical, data) {
		return decoded, fmt.Errorf("%s: non-canonical document", c.Name)
	}
	id, err := contract.Identify(canonical)
	if err != nil {
		return decoded, err
	}
	c.SetIdentity(&value, id)
	return value, nil
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
