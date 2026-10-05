package trainingworkflow

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/densecausal"
	"overgo/internal/gguf"
	"overgo/internal/hfbpe"
	"overgo/internal/strictjson"
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
func trainedInput(input string) (string, error) {
	if info, err := os.Stat(input); err != nil || !info.IsDir() {
		return input, err
	}
	if trained, err := trainingprogram.CheckpointWeightsPath(input); err == nil && isGGUFModelInput(trained) {
		return trained, nil
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	return input, nil
}

// loadTrainableModel materializes the trainable F32 inventory from either
// model representation through its own loader.
func loadTrainableModel(input string) (*densecausal.Model, error) {
	input, err := trainedInput(input)
	if err != nil {
		return nil, err
	}
	if isGGUFModelInput(input) {
		return densecausal.LoadGGUF(input)
	}
	return densecausal.Load(input)
}

// loadFrozenModel materializes the trainable model from a GGUF base holding the
// weights hold names as the file stores them, each dequantized as a layer reads it
// (densecausal.LoadGGUFFrozen); a base that is not a GGUF has nothing stored to hold.
func loadFrozenModel(input string, hold func(name string) bool) (*densecausal.Model, error) {
	input, err := trainedInput(input)
	if err != nil {
		return nil, err
	}
	if !isGGUFModelInput(input) {
		return nil, errors.New("training workflow: weights stay quantized only in a GGUF base")
	}
	return densecausal.LoadGGUFFrozen(input, hold)
}

// trainableTokenizer: the encoder training reads text with, each token's
// text for showing what it read, and the end-of-sequence token the model
// declares, which separates documents packed into one training window;
// ended is false when the model declares none.
type trainableTokenizer struct {
	encode func(string) ([]int, error)
	piece  func(int) string
	end    int
	ended  bool
}

// loadTrainableEncoder resolves the text encoder beside the weights.
func loadTrainableEncoder(input string) (func(string) ([]int, error), error) {
	loaded, err := loadTrainableTokenizer(input)
	return loaded.encode, err
}

// loadTrainableTokenizer resolves the tokenizer beside the weights: the HF
// tokenizer files in a safetensors directory, with the end-of-sequence
// token its configs declare, or the tokenizer GGUF metadata embeds.
func loadTrainableTokenizer(input string) (trainableTokenizer, error) {
	return loadTrainableTokenizerFor(input, false)
}

// loadTrainableTokenizerFor is loadTrainableTokenizer, reading chat text when
// chat is set: a GGUF vocabulary parses the control tokens in the text and adds
// its start token, as serving encodes a chat prompt.
func loadTrainableTokenizerFor(input string, chat bool) (trainableTokenizer, error) {
	input, err := trainedInput(input)
	if err != nil {
		return trainableTokenizer{}, err
	}
	if !isGGUFModelInput(input) {
		loaded, err := hfbpe.Load(input)
		if err != nil {
			return trainableTokenizer{}, err
		}
		end, ended, err := declaredEnd(input)
		if err != nil {
			return trainableTokenizer{}, err
		}
		return trainableTokenizer{encode: loaded.Encode, piece: func(id int) string { return loaded.Decode([]int{id}) }, end: end, ended: ended}, nil
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
	encodeWith := func(options tokenizer.EncodeOptions) func(string) ([]int, error) {
		return func(text string) ([]int, error) {
			ids, err := vocabulary.Encode(text, options)
			if err != nil {
				return nil, err
			}
			tokens := make([]int, len(ids))
			for index, id := range ids {
				tokens[index] = int(id)
			}
			return tokens, nil
		}
	}
	return trainableTokenizer{
		encode: encodeWith(tokenizer.EncodeOptions{AddSpecial: chat, ParseSpecial: chat}),
		// A piece the vocabulary cannot decode shows empty; the id stays beside it.
		piece: func(id int) string { piece, _ := vocabulary.DecodePiece(tokenizer.TokenID(id), true); return piece },
		end:   int(vocabulary.EOS), ended: vocabulary.EOS != tokenizer.NullToken,
	}, nil
}

// declaredEnd reads the end-of-sequence token a safetensors directory
// declares: the first of generation_config.json's eos_token_id, else of
// config.json's, the precedence the HF converter resolves. A trained
// checkpoint carries config.json alone, so its declaration still reads.
func declaredEnd(directory string) (int, bool, error) {
	for _, name := range []string{"generation_config.json", "config.json"} {
		data, err := os.ReadFile(filepath.Join(directory, name))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return 0, false, err
		}
		var declared struct {
			EOS json.RawMessage `json:"eos_token_id"`
		}
		if err := json.Unmarshal(data, &declared); err != nil {
			return 0, false, fmt.Errorf("training workflow: %s: %w", name, err)
		}
		if !strictjson.HasValue(declared.EOS) {
			continue
		}
		var ids []int
		if json.Unmarshal(declared.EOS, &ids) != nil {
			var id int
			if err := json.Unmarshal(declared.EOS, &id); err != nil {
				return 0, false, fmt.Errorf("training workflow: %s eos_token_id is neither a token nor a list of tokens", name)
			}
			ids = []int{id}
		}
		if len(ids) != 0 {
			return ids[0], true, nil
		}
	}
	return 0, false, nil
}
