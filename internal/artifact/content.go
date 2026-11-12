package artifact

import (
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
)

const MaxContentBytes = 64 << 20

// IdentifyFile gives the identity of the bytes of the file at path, read
// whole: what names a file by its content where its place, size and time,
// which other bytes can have, would not.
func IdentifyFile(kind Kind, path string) (ID, error) {
	file, err := os.Open(path)
	if err != nil {
		return ID{}, err
	}
	defer file.Close()
	id, _, err := Identify(kind, file)
	return id, err
}

// ReadContentFile reads a nonempty regular file within the inline content bound.
// The publisher still owns content identity and schema validation.
func ReadContentFile(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxContentBytes {
		return nil, errors.New("artifact: invalid content file size or mode")
	}
	data, err := io.ReadAll(io.LimitReader(file, MaxContentBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) == 0 || len(data) > MaxContentBytes {
		return nil, errors.New("artifact: content file changed beyond size bound")
	}
	return data, nil
}

// Content defines bounded inline content-addressed document.
type Content struct {
	Descriptor Descriptor `json:"descriptor"`
	Data       []byte     `json:"data"`
}

func (c Content) Validate() error {
	if err := c.Descriptor.Validate(); err != nil {
		return err
	}
	// Each bound is said apart, with the sizes: one message for all three left a failed draw of MOSS's reading
	// (partiprompt_592, 2026-11-11) without its cause.
	switch {
	case len(c.Data) == 0:
		return errors.New("artifact: content is empty")
	case len(c.Data) > MaxContentBytes:
		return fmt.Errorf("artifact: content is %d bytes, over the inline limit of %d", len(c.Data), MaxContentBytes)
	case c.Descriptor.Size != uint64(len(c.Data)):
		return fmt.Errorf("artifact: content is %d bytes, and its descriptor says %d", len(c.Data), c.Descriptor.Size)
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
