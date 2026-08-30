package artifact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

const (
	InitialDocumentVersion uint16 = iota + 1
	SecondDocumentVersion
)

// DocumentContract defines inline content identity and storage schema.
type DocumentContract struct {
	Kind      Kind
	MediaType string
	Schema    string
}

// Validate checks the exact stored-document contract.
func (c DocumentContract) Validate() error {
	if c.Kind == KindInvalid || int(c.Kind) >= len(kindNames) || c.MediaType == "" || c.Schema == "" ||
		strings.TrimSpace(c.MediaType) != c.MediaType || strings.ContainsAny(c.MediaType, "\r\n") ||
		strings.TrimSpace(c.Schema) != c.Schema || strings.ContainsAny(c.Schema, "\r\n") {
		return errors.New("artifact: invalid document contract")
	}
	return nil
}

// ReadContent materializes one bounded content stream.
func ReadContent(ctx context.Context, reader Reader, id ID) (Content, bool, error) {
	if ctx == nil || reader == nil {
		return Content{}, false, errors.New("artifact: nil content context or reader")
	}
	descriptor, stream, found, err := reader.OpenContent(ctx, id)
	if err != nil || !found {
		return Content{}, found, err
	}
	content, err := ReadContentFrom(descriptor, stream)
	return content, err == nil, err
}

// RequireTypedContent returns one immutable content body only when its
// descriptor declares the media and schema needed for typed replay.
func RequireTypedContent(ctx context.Context, reader Reader, id ID) (Content, error) {
	content, found, err := ReadContent(ctx, reader, id)
	if err != nil {
		return Content{}, err
	}
	if !found || content.Descriptor.MediaType == "" || content.Descriptor.Schema == "" {
		return Content{}, fmt.Errorf("artifact: typed content is absent: %s", id)
	}
	return content, nil
}

// ContentVisitor is implemented by repositories that stream many
// contents under one storage acquisition; ReadContents rides it when
// present and falls back to per-id reads otherwise.
type ContentVisitor interface {
	VisitContents(ctx context.Context, ids []ID, visit func(Descriptor, io.Reader) error) error
}

// ContentPresence is implemented by repositories that answer content
// presence for many ids under one state acquisition.
type ContentPresence interface {
	PresentContents(ctx context.Context, ids []ID) ([]ID, error)
}

// ReadContents materializes every id and visits the results in caller
// order, whatever order the storage layer streams them in. Identity,
// byte bounds, and cancellation follow ReadContent exactly; an absent
// id is the storage layer's exact missing-content error, never a
// silent skip.
func ReadContents(ctx context.Context, reader Reader, ids []ID, visit func(Content) error) error {
	if ctx == nil || reader == nil || visit == nil {
		return errors.New("artifact: nil batch read context, reader, or visitor")
	}
	visitor, batched := reader.(ContentVisitor)
	if !batched {
		for _, id := range ids {
			content, found, err := ReadContent(ctx, reader, id)
			if err != nil {
				return err
			}
			if !found {
				return fmt.Errorf("artifact: content is absent: %s", id)
			}
			if err := visit(content); err != nil {
				return err
			}
		}
		return nil
	}
	loaded := make(map[ID]Content, len(ids))
	err := visitor.VisitContents(ctx, ids, func(descriptor Descriptor, stream io.Reader) error {
		content, err := ReadContentFrom(descriptor, stream)
		if err != nil {
			return err
		}
		loaded[descriptor.ID] = content
		return nil
	})
	if err != nil {
		return err
	}
	for _, id := range ids {
		content, found := loaded[id]
		if !found {
			return fmt.Errorf("artifact: content is absent: %s", id)
		}
		if err := visit(content); err != nil {
			return err
		}
	}
	return nil
}

// ReadContentFrom materializes one opened content stream.
func ReadContentFrom(descriptor Descriptor, stream io.Reader) (Content, error) {
	if stream == nil {
		return Content{}, errors.New("artifact: nil content stream")
	}
	if descriptor.Size == 0 || descriptor.Size > MaxContentBytes {
		return Content{}, errors.New("artifact: invalid content descriptor size")
	}
	data, err := io.ReadAll(io.LimitReader(stream, int64(descriptor.Size)+1))
	if err != nil {
		return Content{}, err
	}
	content := Content{Descriptor: descriptor, Data: data}
	if err := content.Validate(); err != nil {
		return Content{}, err
	}
	return content, nil
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

// ContentJSON marshals value canonically and binds it to id under
// the contract. Every typed document family publishes through this
// one owner; copying the marshal-then-Content pair is a census hit.
func (c DocumentContract) ContentJSON(id ID, value any) (Content, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return Content{}, fmt.Errorf("artifact: encode %s document: %w", c.Schema, err)
	}
	return c.Content(id, data)
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
	if err := c.Validate(); err != nil {
		return err
	}
	if len(data) == 0 || len(data) > MaxContentBytes {
		return errors.New("artifact: invalid document contract or size")
	}
	return nil
}
