package trainingworkflow

import (
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/densecausal"
	"overgo/internal/gguf"
	"overgo/internal/hfbpe"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingprogram"
)

// isGGUFModelInput reports whether the model input names one GGUF file.
// GGUF is a different representation of the same weights, so the workflow
// accepts it wherever a safetensors directory is accepted.
func isGGUFModelInput(input string) bool {
	info, err := os.Stat(input)
	return err == nil && !info.IsDir() && strings.EqualFold(filepath.Ext(input), ".gguf")
}

// trainedInput resolves a model input that names a run's output directory
// to the trained model the run published there over a GGUF base.
func trainedInput(input string) string {
	if trained := filepath.Join(input, trainingprogram.CheckpointModel); isGGUFModelInput(trained) {
		return trained
	}
	return input
}

// loadTrainableModel materializes the trainable F32 inventory from either
// model representation through its own loader.
func loadTrainableModel(input string) (*densecausal.Model, error) {
	if input = trainedInput(input); isGGUFModelInput(input) {
		return densecausal.LoadGGUF(input)
	}
	return densecausal.Load(input)
}

// trainableTokenizer: the encoder training reads text with, and each token's
// text for showing what it read.
type trainableTokenizer struct {
	encode func(string) ([]int, error)
	piece  func(int) string
}

// loadTrainableEncoder resolves the text encoder beside the weights.
func loadTrainableEncoder(input string) (func(string) ([]int, error), error) {
	loaded, err := loadTrainableTokenizer(input)
	return loaded.encode, err
}

// loadTrainableTokenizer resolves the tokenizer beside the weights: the HF
// tokenizer files in a safetensors directory, or the tokenizer GGUF
// metadata embeds.
func loadTrainableTokenizer(input string) (trainableTokenizer, error) {
	if !isGGUFModelInput(input) {
		loaded, err := hfbpe.Load(input)
		if err != nil {
			return trainableTokenizer{}, err
		}
		return trainableTokenizer{encode: loaded.Encode, piece: func(id int) string { return loaded.Decode([]int{id}) }}, nil
	}
	file, err := gguf.Open(input)
	if err != nil {
		return trainableTokenizer{}, err
	}
	defer file.Close()
	vocabulary, err := tokenizer.Load(file)
	if err != nil {
		return trainableTokenizer{}, err
	}
	return trainableTokenizer{
		encode: func(text string) ([]int, error) {
			ids, err := vocabulary.Encode(text, tokenizer.EncodeOptions{})
			if err != nil {
				return nil, err
			}
			tokens := make([]int, len(ids))
			for index, id := range ids {
				tokens[index] = int(id)
			}
			return tokens, nil
		},
		// A piece the vocabulary cannot decode shows empty; the id stays beside it.
		piece: func(id int) string { piece, _ := vocabulary.DecodePiece(tokenizer.TokenID(id), true); return piece },
	}, nil
}
