// Model-config components as typed store artifacts: the inference- and
// training-relevant declarations a checkpoint directory carries, committed
// beside the model so generic code reads declarations, never literals.
package modelartifact

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

const (
	ModelConfigVersion   uint16 = 1
	ModelConfigMediaType        = "application/vnd.overgo.model-config+json"
	ModelConfigSchema           = "overgo/model-config/v1"
)

// SequenceExtensionConfig is the declared k-mer tokenizer extension: chunk
// width, ID range, the special tokens in declaration order, and whether
// untagged input auto-wraps. The same facts the converter writes into
// tokenizer metadata, preserved as a store declaration.
type SequenceExtensionConfig struct {
	K             uint32   `json:"k"`
	StartID       uint32   `json:"start_id"`
	Vocabulary    uint32   `json:"vocabulary"`
	SpecialTokens []string `json:"special_tokens"`
	AutoTags      bool     `json:"auto_tags"`
}

// GenerationEssentials are the generation-relevant declarations: begin and
// end-of-sequence token IDs, the declared context length, and the
// sampling the model's own configuration or card recommends.
type GenerationEssentials struct {
	BOSTokens     []int64             `json:"bos_tokens,omitempty"`
	EOSTokens     []int64             `json:"eos_tokens,omitempty"`
	ContextLength uint64              `json:"context_length,omitzero"`
	Sampling      *GenerationSampling `json:"sampling,omitempty"`
}

// GenerationSampling is the sampling a model ships with: the
// generation_config.json of its checkpoint, or the settings its model
// card recommends when the checkpoint declares none. Origin names which;
// a model card declaration carries its URL and mode. Zero fields are
// undeclared and leave the runtime policy's value standing.
type GenerationSampling struct {
	Origin            string  `json:"origin"`
	Mode              string  `json:"mode,omitzero"`
	DoSample          bool    `json:"do_sample"`
	Temperature       float64 `json:"temperature,omitzero"`
	TopP              float64 `json:"top_p,omitzero"`
	TopK              int     `json:"top_k,omitzero"`
	MinP              float64 `json:"min_p,omitzero"`
	PresencePenalty   float64 `json:"presence_penalty,omitzero"`
	RepetitionPenalty float64 `json:"repetition_penalty,omitzero"`
}

// declaredSampling is the shape both generation_config.json and a
// model_card_sampling.json declaration share.
type declaredSampling struct {
	Source            string   `json:"source"`
	Mode              string   `json:"mode"`
	DoSample          *bool    `json:"do_sample"`
	Temperature       *float64 `json:"temperature"`
	TopP              *float64 `json:"top_p"`
	TopK              *int     `json:"top_k"`
	MinP              *float64 `json:"min_p"`
	PresencePenalty   *float64 `json:"presence_penalty"`
	RepetitionPenalty *float64 `json:"repetition_penalty"`
}

// ModelCardSamplingFile is the declaration file an operator writes beside
// a checkpoint (or in a directory of its own for a GGUF-only model) to
// record the sampling a model card recommends, with its source URL.
const ModelCardSamplingFile = "model_card_sampling.json"

func (declared declaredSampling) sampling(origin string) *GenerationSampling {
	if declared.DoSample == nil && declared.Temperature == nil && declared.TopP == nil && declared.TopK == nil &&
		declared.MinP == nil && declared.PresencePenalty == nil && declared.RepetitionPenalty == nil {
		return nil
	}
	result := &GenerationSampling{Origin: origin, Mode: declared.Mode}
	if declared.Source != "" {
		result.Origin = declared.Source
	}
	if declared.DoSample != nil {
		result.DoSample = *declared.DoSample
	}
	if declared.Temperature != nil {
		result.Temperature = *declared.Temperature
	}
	if declared.TopP != nil {
		result.TopP = *declared.TopP
	}
	if declared.TopK != nil {
		result.TopK = *declared.TopK
	}
	if declared.MinP != nil {
		result.MinP = *declared.MinP
	}
	if declared.PresencePenalty != nil {
		result.PresencePenalty = *declared.PresencePenalty
	}
	if declared.RepetitionPenalty != nil {
		result.RepetitionPenalty = *declared.RepetitionPenalty
	}
	return result
}

// ConfigSource names one source file and its content digest, so the
// extraction is a claim the original directory can refute.
type ConfigSource struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

// ModelConfigDocument binds the extracted configuration components to a
// model identity with provenance digests for every file consumed.
type ModelConfigDocument struct {
	Version    uint16                   `json:"version"`
	Model      artifact.ID              `json:"model"`
	Sequence   *SequenceExtensionConfig `json:"sequence,omitempty"`
	Generation *GenerationEssentials    `json:"generation,omitempty"`
	Sources    []ConfigSource           `json:"sources"`
	ID         artifact.ID              `json:"-"`
}

var modelConfigCodec = artifact.JSONDocumentCodec(
	"model config", artifact.KindProfile, ModelConfigMediaType, ModelConfigSchema,
	canonicalizeModelConfig,
	func(value ModelConfigDocument) artifact.ID { return value.ID },
	func(value *ModelConfigDocument, id artifact.ID) { value.ID = id },
	func(value ModelConfigDocument) ModelConfigDocument {
		if value.Sequence != nil {
			sequence := *value.Sequence
			sequence.SpecialTokens = slices.Clone(sequence.SpecialTokens)
			value.Sequence = &sequence
		}
		if value.Generation != nil {
			generation := *value.Generation
			generation.BOSTokens = slices.Clone(generation.BOSTokens)
			generation.EOSTokens = slices.Clone(generation.EOSTokens)
			if generation.Sampling != nil {
				sampling := *generation.Sampling
				generation.Sampling = &sampling
			}
			value.Generation = &generation
		}
		value.Sources = slices.Clone(value.Sources)
		return value
	},
)

func NewModelConfigDocument(
	model artifact.ID,
	sequence *SequenceExtensionConfig,
	generation *GenerationEssentials,
	sources []ConfigSource,
) (ModelConfigDocument, error) {
	return modelConfigCodec.New(ModelConfigDocument{
		Version: ModelConfigVersion, Model: model,
		Sequence: sequence, Generation: generation, Sources: slices.Clone(sources),
	})
}

func ParseModelConfigDocument(content []byte) (ModelConfigDocument, error) {
	return modelConfigCodec.Parse(content)
}

func (d ModelConfigDocument) Content() (artifact.Content, error) {
	return modelConfigCodec.Content(d)
}

// Batch commits the declaration with lineage to the model it describes.
func (d ModelConfigDocument) Batch(key string) (artifact.Batch, error) {
	return modelConfigCodec.Batch(key, d, artifact.DependencyLineage(d.ID, d.Model), nil)
}

// ReadModelConfigComponents extracts the typed components from a checkpoint
// directory: the sequence extension declaration, generation essentials from
// the generation and model configs, and a digest for every file consumed.
// Absent files are absent components, never inventions; malformed files are
// errors, never silence.
func ReadModelConfigComponents(directory string) (*SequenceExtensionConfig, *GenerationEssentials, []ConfigSource, error) {
	var sources []ConfigSource
	read := func(name string) ([]byte, bool, error) {
		data, err := os.ReadFile(filepath.Join(directory, name))
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		if err != nil {
			return nil, false, fmt.Errorf("model config: read %s: %w", name, err)
		}
		digest := sha256.Sum256(data)
		sources = append(sources, ConfigSource{Name: name, SHA256: hex.EncodeToString(digest[:])})
		return data, true, nil
	}

	var sequence *SequenceExtensionConfig
	if data, ok, err := read("dna_config.json"); err != nil {
		return nil, nil, nil, err
	} else if ok {
		var declared struct {
			K             uint32   `json:"k"`
			StartID       uint32   `json:"dna_start_id"`
			Vocabulary    uint32   `json:"dna_vocab_size"`
			SpecialTokens []string `json:"dna_special_tokens"`
			AutoTags      bool     `json:"auto_dna_tags"`
		}
		if err := json.Unmarshal(data, &declared); err != nil {
			return nil, nil, nil, fmt.Errorf("model config: parse dna_config.json: %w", err)
		}
		sequence = &SequenceExtensionConfig{
			K: declared.K, StartID: declared.StartID, Vocabulary: declared.Vocabulary,
			SpecialTokens: declared.SpecialTokens, AutoTags: declared.AutoTags,
		}
	}

	generation := &GenerationEssentials{}
	populated := false
	if data, ok, err := read("generation_config.json"); err != nil {
		return nil, nil, nil, err
	} else if ok {
		var declared map[string]json.RawMessage
		if err := json.Unmarshal(data, &declared); err != nil {
			return nil, nil, nil, fmt.Errorf("model config: parse generation_config.json: %w", err)
		}
		bos, err := tokenIDList(declared["bos_token_id"])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("model config: generation_config.json bos_token_id: %w", err)
		}
		eos, err := tokenIDList(declared["eos_token_id"])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("model config: generation_config.json eos_token_id: %w", err)
		}
		generation.BOSTokens, generation.EOSTokens = bos, eos
		populated = populated || len(bos) > 0 || len(eos) > 0
		var sampling declaredSampling
		if err := json.Unmarshal(data, &sampling); err != nil {
			return nil, nil, nil, fmt.Errorf("model config: parse generation_config.json sampling: %w", err)
		}
		if declared := sampling.sampling("generation_config.json"); declared != nil {
			generation.Sampling = declared
			populated = true
		}
	}
	// A model card declaration, when present, is the operator's explicit
	// record of the recommended settings and stands over the checkpoint's
	// generation config.
	if data, ok, err := read(ModelCardSamplingFile); err != nil {
		return nil, nil, nil, err
	} else if ok {
		var sampling declaredSampling
		if err := json.Unmarshal(data, &sampling); err != nil {
			return nil, nil, nil, fmt.Errorf("model config: parse %s: %w", ModelCardSamplingFile, err)
		}
		declared := sampling.sampling(ModelCardSamplingFile)
		if declared == nil || sampling.Source == "" {
			return nil, nil, nil, fmt.Errorf("model config: %s declares no sampling or no source", ModelCardSamplingFile)
		}
		generation.Sampling = declared
		populated = true
	}
	if data, ok, err := read("config.json"); err != nil {
		return nil, nil, nil, err
	} else if ok {
		var declared struct {
			MaxPositionEmbeddings *uint64 `json:"max_position_embeddings"`
		}
		if err := json.Unmarshal(data, &declared); err != nil {
			return nil, nil, nil, fmt.Errorf("model config: parse config.json: %w", err)
		}
		if declared.MaxPositionEmbeddings != nil {
			generation.ContextLength = *declared.MaxPositionEmbeddings
			populated = true
		}
	}
	if !populated {
		generation = nil
	}
	return sequence, generation, sources, nil
}

// tokenIDList accepts a scalar, list or null token-ID declaration.
func tokenIDList(raw json.RawMessage) ([]int64, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var scalar int64
	if err := json.Unmarshal(raw, &scalar); err == nil {
		return []int64{scalar}, nil
	}
	var list []int64
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, errors.New("token IDs must be an integer, a list of integers, or null")
	}
	return list, nil
}

func canonicalizeModelConfig(value *ModelConfigDocument) error {
	if value == nil || value.Version != ModelConfigVersion {
		return errors.New("model config: invalid version")
	}
	if value.Model.Kind() != artifact.KindModel {
		return errors.New("model config: document requires a model identity")
	}
	if value.Sequence == nil && value.Generation == nil {
		return errors.New("model config: document declares no components")
	}
	if len(value.Sources) == 0 {
		return errors.New("model config: document requires source provenance")
	}
	sort.Slice(value.Sources, func(i, j int) bool { return value.Sources[i].Name < value.Sources[j].Name })
	for i, source := range value.Sources {
		if source.Name == "" || strings.ContainsAny(source.Name, "\x00\r\n\\/") {
			return fmt.Errorf("model config: source %d requires a bare file name", i)
		}
		if i > 0 && source.Name == value.Sources[i-1].Name {
			return fmt.Errorf("model config: source %q listed twice", source.Name)
		}
		if len(source.SHA256) != hex.EncodedLen(sha256.Size) || strings.ToLower(source.SHA256) != source.SHA256 {
			return fmt.Errorf("model config: source %q requires a lowercase sha256 digest", source.Name)
		}
		if _, err := hex.DecodeString(source.SHA256); err != nil {
			return fmt.Errorf("model config: source %q digest is not hex", source.Name)
		}
	}
	return nil
}
