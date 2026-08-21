package artifact

import (
	"errors"
	"slices"
)

const (
	InitialDocumentVersion uint16 = iota + 1
	SecondDocumentVersion
)

// DocumentContract: inline content identity and storage schema.
type DocumentContract struct {
	Kind      Kind
	MediaType string
	Schema    string
}

func (c DocumentContract) Identify(data []byte) (ID, error) {
	if err := c.validateData(data); err != nil {
		return ID{}, err
	}
	return IdentifyBytes(c.Kind, data)
}

func (c DocumentContract) ValidateIdentity(id ID, data []byte) error {
	want, err := c.Identify(data)
	if err != nil {
		return err
	}
	if id != want {
		return errors.New("artifact: document identity mismatch")
	}
	return nil
}

func (c DocumentContract) Descriptor(id ID, size uint64) (Descriptor, error) {
	if id.Kind() != c.Kind || size == 0 || size > MaxContentBytes {
		return Descriptor{}, errors.New("artifact: invalid document descriptor")
	}
	descriptor := Descriptor{ID: id, Size: size, MediaType: c.MediaType, Schema: c.Schema}
	if err := descriptor.Validate(); err != nil {
		return Descriptor{}, err
	}
	return descriptor, nil
}

func (c DocumentContract) Content(id ID, data []byte) (Content, error) {
	if err := c.ValidateIdentity(id, data); err != nil {
		return Content{}, err
	}
	descriptor, err := c.Descriptor(id, uint64(len(data)))
	if err != nil {
		return Content{}, err
	}
	return Content{Descriptor: descriptor, Data: slices.Clone(data)}, nil
}

// ContentBytes binds raw document bytes to their derived identity.
func (c DocumentContract) ContentBytes(data []byte) (Content, error) {
	id, err := c.Identify(data)
	if err != nil {
		return Content{}, err
	}
	return c.Content(id, data)
}

// OwnedContentBytes binds bytes without a clone. Caller must not mutate them.
func (c DocumentContract) OwnedContentBytes(data []byte) (Content, error) {
	id, err := c.Identify(data)
	if err != nil {
		return Content{}, err
	}
	descriptor, err := c.Descriptor(id, uint64(len(data)))
	if err != nil {
		return Content{}, err
	}
	return Content{Descriptor: descriptor, Data: data}, nil
}

func (c DocumentContract) ValidateContent(content Content, id ID) error {
	if err := c.validateDescriptor(content.Descriptor, id); err != nil {
		return err
	}
	return content.Validate()
}

// DependencyLineage binds one child to immutable parent authorities.
func DependencyLineage(child ID, parents ...ID) []Lineage {
	lineage := make([]Lineage, len(parents))
	for index, parent := range parents {
		lineage[index] = Lineage{Child: child, Parent: parent, Relation: RelationDependsOn}
	}
	return lineage
}

func (c DocumentContract) validateDescriptor(descriptor Descriptor, id ID) error {
	if id.Kind() != c.Kind || descriptor.ID != id || descriptor.MediaType != c.MediaType ||
		descriptor.Schema != c.Schema {
		return errors.New("artifact: incompatible document content")
	}
	return nil
}

func NewDocumentBatch(
	key string,
	contents []Content,
	lineage []Lineage,
	aliases []AliasBinding,
) (Batch, error) {
	clonedContents := make([]Content, len(contents))
	for index := range contents {
		clonedContents[index] = contents[index].Clone()
	}
	batch := Batch{
		Key: key, Contents: clonedContents, Lineage: slices.Clone(lineage),
		Aliases: CloneAliasBindings(aliases),
	}
	if err := batch.Validate(); err != nil {
		return Batch{}, err
	}
	return batch, nil
}

func (c DocumentContract) validateData(data []byte) error {
	if c.Kind == KindInvalid || int(c.Kind) >= len(kindNames) || c.MediaType == "" || c.Schema == "" ||
		len(data) == 0 || len(data) > MaxContentBytes {
		return errors.New("artifact: invalid document contract or size")
	}
	probe, err := NewID(c.Kind, [digestBytes]byte{})
	if err != nil {
		return err
	}
	_, err = c.Descriptor(probe, uint64(len(data)))
	return err
}
