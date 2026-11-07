package evaluation

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/model"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

// captionEmbeddings is a Hugging Face text model's token embedding table, which a caption's tokens are read from.
const captionEmbeddings = "model.embed_tokens.weight"

// CaptionText is the text model a caption capture and a caption projector's training read, a GGUF file's own or a
// Hugging Face directory's tokenizer file, configuration and embedding table: its width, the token that ends a text,
// a text's tokens, its added tokens read as the tokens they name and none added of the tokenizer's own, and the
// tokens' embedding rows.
type CaptionText struct {
	Width, End int
	encode     func(string) ([]int, error)
	rows       func([]int) ([]float32, error)
	close      func() error
}

// OpenCaptionText reads the text model at path, refusing one whose tokenizer opens a prompt with a beginning token,
// which a caption's sequence would lack.
func OpenCaptionText(path string) (*CaptionText, error) {
	opens := errors.New("caption text: the text model opens a prompt with a beginning token, which its tokenizer adds to no text")
	if strings.EqualFold(filepath.Ext(path), ".gguf") {
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
			return fail(opens)
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
	vocabulary, err := hfbpe.Load(path)
	if err != nil {
		return nil, err
	}
	var config struct {
		End *int `json:"eos_token_id"`
	}
	if err := jsonfile.Decode(filepath.Join(path, "config.json"), &config); err != nil || config.End == nil {
		return nil, errors.Join(err, errors.New("caption text: the text model's configuration names no single end token"))
	}
	var opening struct {
		Beginning bool `json:"add_bos_token"`
	}
	if err := jsonfile.Decode(filepath.Join(path, "tokenizer_config.json"), &opening); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	if opening.Beginning {
		return nil, opens
	}
	source, err := safetensors.OpenSource(path)
	if err != nil {
		return nil, err
	}
	table, found := source.Tensors[captionEmbeddings]
	if err := source.Close(); err != nil || !found || len(table.Shape) != tensor.PairedExtent {
		return nil, errors.Join(err, fmt.Errorf("caption text: %s holds no %s table", path, captionEmbeddings))
	}
	width := int(table.Shape[1])
	return &CaptionText{Width: width, End: *config.End, encode: vocabulary.Encode, close: func() error { return nil },
		rows: func(ids []int) ([]float32, error) {
			return safetensors.ReadTensorRowsF32(path, captionEmbeddings, width, ids)
		},
	}, nil
}

// Encode is text's tokens.
func (t *CaptionText) Encode(text string) ([]int, error) { return t.encode(text) }

// Rows are the embedding rows of ids, Width values each.
func (t *CaptionText) Rows(ids []int) ([]float32, error) { return t.rows(ids) }

// Close releases the text model's file.
func (t *CaptionText) Close() error { return t.close() }
