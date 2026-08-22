package artifact

import (
	"errors"
	"fmt"
	"slices"
)

const MaxContentBytes = 64 << 20

// Content defines bounded inline content-addressed document.
type Content struct {
	Descriptor Descriptor `json:"descriptor"`
	Data       []byte     `json:"data"`
}

func (c Content) Validate() error {
	if err := c.Descriptor.Validate(); err != nil {
		return err
	}
	if len(c.Data) == 0 || len(c.Data) > MaxContentBytes || c.Descriptor.Size != uint64(len(c.Data)) {
		return errors.New("artifact: invalid content size")
	}
	want, err := IdentifyBytes(c.Descriptor.ID.Kind(), c.Data)
	if err != nil {
		return err
	}
	if want != c.Descriptor.ID {
		return fmt.Errorf("artifact: content identity mismatch: have %s want %s", c.Descriptor.ID, want)
	}
	return nil
}

func (c Content) Clone() Content {
	c.Data = slices.Clone(c.Data)
	return c
}
