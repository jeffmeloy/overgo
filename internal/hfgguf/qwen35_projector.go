package hfgguf

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
)

const (
	qwen35VisionNormEpsilon   = 1e-6
	qwen35VisionChannels      = 3
	qwen35TemporalPatchSlices = 2
)

var qwen35VisionLayerPattern = regexp.MustCompile(`^model\.visual\.blocks\.(\d+)\.(.+)$`)

type qwen35VisionConfig struct {
	Depth             uint32  `json:"depth"`
	HiddenActivation  string  `json:"hidden_act"`
	HiddenSize        uint32  `json:"hidden_size"`
	InChannels        uint32  `json:"in_channels"`
	InitializerRange  float32 `json:"initializer_range"`
	IntermediateSize  uint32  `json:"intermediate_size"`
	ModelType         string  `json:"model_type"`
	Heads             uint32  `json:"num_heads"`
	Positions         uint32  `json:"num_position_embeddings"`
	OutputHidden      uint32  `json:"out_hidden_size"`
	PatchSize         uint32  `json:"patch_size"`
	MergeSize         uint32  `json:"spatial_merge_size"`
	TemporalPatchSize uint32  `json:"temporal_patch_size"`
	RopeTheta         float32 `json:"rope_theta"`
	DeepstackIndexes  []int   `json:"deepstack_visual_indexes"`
}

type qwen35VisionProcessor struct {
	Size struct {
		LongestEdge  uint32 `json:"longest_edge"`
		ShortestEdge uint32 `json:"shortest_edge"`
	} `json:"size"`
	PatchSize         uint32    `json:"patch_size"`
	TemporalPatchSize uint32    `json:"temporal_patch_size"`
	MergeSize         uint32    `json:"merge_size"`
	ImageMean         []float32 `json:"image_mean"`
	ImageStd          []float32 `json:"image_std"`
	ProcessorClass    string    `json:"processor_class"`
	ProcessorType     string    `json:"image_processor_type"`
}

// Qwen35ProjectorConversion: typed Qwen 3.5 vision artifact.
func Qwen35ProjectorConversion(repository *hfrepo.Repository) ([]gguf.Metadata, []gguf.TensorData, error) {
	if repository == nil || repository.Tensors == nil || repository.Identity.ModelType != "qwen3_5" {
		return nil, nil, errors.New("HF/GGUF adapter: invalid Qwen 3.5 projector repository")
	}
	var config qwen35VisionConfig
	if err := strictjson.DecodeBytes(repository.Config["vision_config"], &config); err != nil {
		return nil, nil, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 vision config: %w", err)
	}
	var processor qwen35VisionProcessor
	file, err := os.Open(filepath.Join(repository.Directory, "preprocessor_config.json"))
	if err != nil {
		return nil, nil, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 processor: %w", err)
	}
	decodeErr := strictjson.DecodeBounded(file, 1<<20, &processor)
	closeErr := file.Close()
	if decodeErr != nil || closeErr != nil {
		return nil, nil, errors.Join(decodeErr, closeErr)
	}
	if err := validateQwen35VisionConfig(config, processor); err != nil {
		return nil, nil, err
	}
	side := uint32(math.Sqrt(float64(config.Positions)))
	metadata := []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "clip"),
		metadata("general.name", gguf.ValueTypeString, filepath.Base(repository.Directory)+" multimodal projector"),
		metadata("clip.projector_type", gguf.ValueTypeString, "qwen3vl_merger"),
		metadata("clip.has_vision_encoder", gguf.ValueTypeBool, true),
		metadata("clip.use_gelu", gguf.ValueTypeBool, true),
		metadata("clip.vision.image_size", gguf.ValueTypeUint32, side*config.PatchSize),
		metadata("clip.vision.patch_size", gguf.ValueTypeUint32, config.PatchSize),
		metadata("clip.vision.embedding_length", gguf.ValueTypeUint32, config.HiddenSize),
		metadata("clip.vision.feed_forward_length", gguf.ValueTypeUint32, config.IntermediateSize),
		metadata("clip.vision.projection_dim", gguf.ValueTypeUint32, config.OutputHidden),
		metadata("clip.vision.block_count", gguf.ValueTypeUint32, config.Depth),
		metadata("clip.vision.attention.head_count", gguf.ValueTypeUint32, config.Heads),
		metadata("clip.vision.spatial_merge_size", gguf.ValueTypeUint32, config.MergeSize),
		metadata("clip.vision.rope.freq_base", gguf.ValueTypeFloat32, config.RopeTheta),
		metadata("clip.vision.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(qwen35VisionNormEpsilon)),
		gguf.ArrayMetadata("clip.vision.image_mean", gguf.ValueTypeFloat32, processor.ImageMean),
		gguf.ArrayMetadata("clip.vision.image_std", gguf.ValueTypeFloat32, processor.ImageStd),
	}
	tensors, err := qwen35ProjectorTensors(repository.Tensors, config)
	if err != nil {
		return nil, nil, err
	}
	return metadata, tensors, nil
}

func validateQwen35VisionConfig(config qwen35VisionConfig, processor qwen35VisionProcessor) error {
	side := uint32(math.Sqrt(float64(config.Positions)))
	if config.ModelType != "qwen3_5" || config.HiddenActivation != "gelu_pytorch_tanh" || config.InChannels != qwen35VisionChannels ||
		config.Depth == 0 || config.HiddenSize == 0 || config.IntermediateSize == 0 || config.Heads == 0 ||
		config.Positions == 0 || side*side != config.Positions || config.OutputHidden == 0 || config.PatchSize == 0 ||
		config.MergeSize == 0 || config.TemporalPatchSize != qwen35TemporalPatchSlices || config.RopeTheta <= 0 || config.HiddenSize%config.Heads != 0 ||
		processor.PatchSize != config.PatchSize || processor.TemporalPatchSize != config.TemporalPatchSize ||
		processor.MergeSize != config.MergeSize || len(config.DeepstackIndexes) != 0 ||
		len(processor.ImageMean) != qwen35VisionChannels || len(processor.ImageStd) != qwen35VisionChannels {
		return errors.New("HF/GGUF adapter: Qwen 3.5 vision configuration is unsupported")
	}
	for channel := range processor.ImageStd {
		if processor.ImageStd[channel] <= 0 || !finiteFloat32(processor.ImageMean[channel]) || !finiteFloat32(processor.ImageStd[channel]) {
			return fmt.Errorf("HF/GGUF adapter: Qwen 3.5 processor channel %d is invalid", channel)
		}
	}
	return nil
}

func finiteFloat32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func qwen35ProjectorTensors(source *safetensors.Source, config qwen35VisionConfig) ([]gguf.TensorData, error) {
	result := make([]gguf.TensorData, 0, 2+int(config.Depth)*12)
	for _, sourceName := range source.Names() {
		if !strings.HasPrefix(sourceName, "model.visual.") {
			continue
		}
		tensor := source.Tensors[sourceName]
		if tensor.DType != "BF16" {
			return nil, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 projector tensor %q uses %s", sourceName, tensor.DType)
		}
		if sourceName == "model.visual.patch_embed.proj.weight" {
			parts, err := qwen35TemporalPatchTensors(tensor, config)
			if err != nil {
				return nil, err
			}
			result = append(result, parts...)
			continue
		}
		destination, ok := qwen35ProjectorTensorName(sourceName)
		if !ok {
			return nil, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 projector tensor %q has no mapping", sourceName)
		}
		result = append(result, gguf.TensorData{
			Name: destination, Shape: gguf.ReverseShape(tensor.Shape), Type: gguf.DTypeBF16, Data: tensor.Reader(),
		})
	}
	sort.Slice(result, func(left, right int) bool { return result[left].Name < result[right].Name })
	return result, nil
}

func qwen35ProjectorTensorName(source string) (string, bool) {
	fixed := map[string]string{
		"model.visual.patch_embed.proj.bias":    "v.patch_embd.bias",
		"model.visual.pos_embed.weight":         "v.position_embd.weight",
		"model.visual.merger.norm.weight":       "v.post_ln.weight",
		"model.visual.merger.norm.bias":         "v.post_ln.bias",
		"model.visual.merger.linear_fc1.weight": "mm.0.weight",
		"model.visual.merger.linear_fc1.bias":   "mm.0.bias",
		"model.visual.merger.linear_fc2.weight": "mm.2.weight",
		"model.visual.merger.linear_fc2.bias":   "mm.2.bias",
	}
	if destination, ok := fixed[source]; ok {
		return destination, true
	}
	match := qwen35VisionLayerPattern.FindStringSubmatch(source)
	if match == nil {
		return "", false
	}
	suffixes := map[string]string{
		"attn.qkv.weight": "attn_qkv.weight", "attn.qkv.bias": "attn_qkv.bias",
		"attn.proj.weight": "attn_out.weight", "attn.proj.bias": "attn_out.bias",
		"mlp.linear_fc1.weight": "ffn_up.weight", "mlp.linear_fc1.bias": "ffn_up.bias",
		"mlp.linear_fc2.weight": "ffn_down.weight", "mlp.linear_fc2.bias": "ffn_down.bias",
		"norm1.weight": "ln1.weight", "norm1.bias": "ln1.bias",
		"norm2.weight": "ln2.weight", "norm2.bias": "ln2.bias",
	}
	destination, ok := suffixes[match[2]]
	if !ok {
		return "", false
	}
	if _, err := strconv.Atoi(match[1]); err != nil {
		return "", false
	}
	return "v.blk." + match[1] + "." + destination, true
}

func qwen35TemporalPatchTensors(tensor safetensors.Tensor, config qwen35VisionConfig) ([]gguf.TensorData, error) {
	want := []uint64{uint64(config.HiddenSize), qwen35VisionChannels, uint64(config.TemporalPatchSize), uint64(config.PatchSize), uint64(config.PatchSize)}
	if len(tensor.Shape) != len(want) {
		return nil, errors.New("HF/GGUF adapter: Qwen 3.5 temporal patch rank is invalid")
	}
	for index := range want {
		if tensor.Shape[index] != want[index] {
			return nil, fmt.Errorf("HF/GGUF adapter: Qwen 3.5 temporal patch shape is %v", tensor.Shape)
		}
	}
	shape := []uint64{uint64(config.PatchSize), uint64(config.PatchSize), qwen35VisionChannels, uint64(config.HiddenSize)}
	result := make([]gguf.TensorData, qwen35TemporalPatchSlices)
	for temporal := range result {
		name := "v.patch_embd.weight"
		if temporal == 1 {
			name += ".1"
		}
		result[temporal] = gguf.TensorData{
			Name:  name,
			Shape: shape, Type: gguf.DTypeBF16, Data: qwen35TemporalPatchReader(tensor, temporal, config),
		}
	}
	return result, nil
}

func qwen35TemporalPatchReader(tensor safetensors.Tensor, selected int, config qwen35VisionConfig) io.Reader {
	reader, writer := io.Pipe()
	go func() {
		defer writer.Close()
		planeBytes := int(config.PatchSize * config.PatchSize * 2)
		planes := [][]byte{make([]byte, planeBytes), make([]byte, planeBytes)}
		source := tensor.Reader()
		for output := uint32(0); output < config.HiddenSize; output++ {
			for channel := 0; channel < 3; channel++ {
				for temporal := range planes {
					if _, err := io.ReadFull(source, planes[temporal]); err != nil {
						_ = writer.CloseWithError(err)
						return
					}
				}
				if _, err := writer.Write(planes[selected]); err != nil {
					return
				}
			}
		}
	}()
	return reader
}
