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
	value = c.clone(value)
	c.SetIdentity(&value, ID{})
	if err := c.Canonicalize(&value); err != nil {
		return value, err
	}
	data, err := c.Encode(value)
	if err != nil {
		return value, err
	}
	id, err := c.Contract.Identify(data)
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
	value, err := c.New(decoded)
	if err != nil {
		return decoded, err
	}
	canonical, err := c.ContentBytes(value)
	if err != nil {
		return decoded, err
	}
	if !bytes.Equal(canonical, data) {
		return decoded, fmt.Errorf("%s: non-canonical document", c.Name)
	}
	return value, nil
}

func (c DocumentCodec[T]) ValidateIdentity(value T) error {
	if err := c.validate(); err != nil {
		return err
	}
	id := c.Identity(value)
	if id.Kind() != c.Contract.Kind {
		return fmt.Errorf("%s: invalid document identity", c.Name)
	}
	canonical := c.clone(value)
	c.SetIdentity(&canonical, ID{})
	if err := c.Canonicalize(&canonical); err != nil {
		return err
	}
	c.SetIdentity(&canonical, id)
	if !reflect.DeepEqual(value, canonical) {
		return fmt.Errorf("%s: document is not canonical", c.Name)
	}
	c.SetIdentity(&canonical, ID{})
	data, err := c.Encode(canonical)
	if err != nil {
		return err
	}
	if err := c.Contract.ValidateIdentity(id, data); err != nil {
		return fmt.Errorf("%s: document identity mismatch", c.Name)
	}
	return nil
}

func (c DocumentCodec[T]) ContentBytes(value T) ([]byte, error) {
	if err := c.ValidateIdentity(value); err != nil {
		return nil, err
	}
	canonical := c.clone(value)
	c.SetIdentity(&canonical, ID{})
	return c.Encode(canonical)
}

func (c DocumentCodec[T]) Content(value T) (Content, error) {
	id := c.Identity(value)
	data, err := c.ContentBytes(value)
	if err != nil {
		return Content{}, err
	}
	return c.Contract.Content(id, data)
}

func (c DocumentCodec[T]) clone(value T) T {
	if c.Clone != nil {
		return c.Clone(value)
	}
	return value
}

func (c DocumentCodec[T]) validate() error {
	if c.Name == "" || c.Decode == nil || c.Encode == nil || c.Canonicalize == nil ||
		c.Identity == nil || c.SetIdentity == nil {
		return errors.New("artifact: incomplete document codec")
	}
	return nil
}
