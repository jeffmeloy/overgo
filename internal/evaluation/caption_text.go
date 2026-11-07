package evaluation

import (
	"context"
	"errors"

	"overgo/internal/gguf"
	// The directory opener reads a Hugging Face directory as the GGUF its conversion writes.
	_ "overgo/internal/hfconvert"
	"overgo/internal/model"
	"overgo/internal/tokenizer"
)

// CaptionText is the text model a caption capture and a caption projector's training read, a GGUF file or a Hugging
// Face directory, both through gguf.Open: its width, the token that ends a text, a text's tokens, its added tokens read
// as the tokens they name and none added of the tokenizer's own, and the tokens' embedding rows.
type CaptionText struct {
	Width, End int
	encode     func(string) ([]int, error)
	rows       func([]int) ([]float32, error)
	close      func() error
}

// OpenCaptionText reads the text model at path, refusing one whose tokenizer opens a prompt with a beginning token,
// which a caption's sequence would lack.
func OpenCaptionText(path string) (*CaptionText, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (*CaptionText, error) { return nil, errors.Join(cause, file.Close()) }
	vocabulary, err := tokenizer.Load(file)
	if err != nil {
		return fail(err)
	}
	if vocabulary.AddBOS {
		return fail(errors.New("caption text: the text model opens a prompt with a beginning token, which its tokenizer adds to no text"))
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return fail(err)
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return fail(err)
	}
	return &CaptionText{Width: int(spec.EmbeddingLength), End: int(vocabulary.EOS), close: file.Close,
		encode: func(text string) ([]int, error) {
			ids, err := vocabulary.Encode(text, tokenizer.EncodeOptions{ParseSpecial: true})
			tokens := make([]int, len(ids))
			for index, id := range ids {
				tokens[index] = int(id)
			}
			return tokens, err
		},
		rows: func(ids []int) ([]float32, error) {
			tokens := make([]uint32, len(ids))
			for index, id := range ids {
				tokens[index] = uint32(id)
			}
			rows, err := model.LoadHostRows(context.Background(), file, weights.TokenEmbedding, tokens)
			return rows.Data, err
		},
	}, nil
}

// Encode is text's tokens.
func (t *CaptionText) Encode(text string) ([]int, error) { return t.encode(text) }

// Rows are the embedding rows of ids, Width values each.
func (t *CaptionText) Rows(ids []int) ([]float32, error) { return t.rows(ids) }

// Close releases the text model's file.
func (t *CaptionText) Close() error { return t.close() }
