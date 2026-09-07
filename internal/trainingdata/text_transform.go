package trainingdata

import (
	"context"
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"

	"overgo/internal/artifact"
)

// TextTransform declares target-text case handling without changing the raw
// source record. The Unicode table version participates in its identity.
type TextTransform struct {
	Version   uint16      `json:"version"`
	Lowercase bool        `json:"lowercase"`
	Unicode   string      `json:"unicode"`
	ID        artifact.ID `json:"-"`
}

var textTransformCodec = artifact.JSONDocumentCodec(
	"training text transform", artifact.KindProfile, artifact.JSONMediaType, "overgo/training-text-transform/v1",
	func(value *TextTransform) error {
		if value.Version != artifact.InitialDocumentVersion || value.Unicode != unicode.Version {
			return errors.New("training text transform: version or Unicode tables differ")
		}
		return nil
	},
	func(value TextTransform) artifact.ID { return value.ID },
	func(value *TextTransform, id artifact.ID) { value.ID = id },
	func(value TextTransform) TextTransform { return value },
)

// NewTextTransform seals explicit preservation or Unicode lowercase handling.
func NewTextTransform(lowercase bool) (TextTransform, error) {
	return textTransformCodec.NewInitial(TextTransform{Lowercase: lowercase, Unicode: unicode.Version})
}

// RequireTextTransform loads and validates an exact stored transformation.
func RequireTextTransform(ctx context.Context, reader artifact.Reader, id artifact.ID) (TextTransform, error) {
	return textTransformCodec.Require(ctx, reader, id)
}

// Content returns the canonical transformation artifact.
func (transform TextTransform) Content() (artifact.Content, error) {
	return textTransformCodec.Content(transform)
}

// Apply transforms a target without altering its source or silently repairing
// malformed UTF-8. Tokenization remains the tokenizer owner's responsibility.
func (transform TextTransform) Apply(text string) (string, error) {
	if err := textTransformCodec.ValidateIdentity(transform); err != nil {
		return "", err
	}
	if !utf8.ValidString(text) {
		return "", errors.New("training text transform: invalid UTF-8")
	}
	if transform.Lowercase {
		return strings.ToLower(text), nil
	}
	return text, nil
}
