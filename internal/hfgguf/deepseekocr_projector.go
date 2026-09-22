package hfgguf

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/projector"
)

// The DeepSeek-OCR document encoder: a SAM tower whose windowed and global
// blocks reduce the page to a token grid, a CLIP tower over that grid, and a
// linear projector into the decoder's width, with the newline and view
// separator embeddings the prompt program places between rows and views.
// The mmproj carries them under the names the projector runtime reads.
const (
	deepSeekOCRProjectorType = "deepseekocr"
	deepSeekOCRVisionConfig  = "vision_config"
	deepSeekOCRProcessorFile = "processor_config.json"
	// deepSeekOCRDefaultVisionEpsilon is CLIP's layer norm epsilon, which the
	// encoder leaves implicit.
	deepSeekOCRDefaultVisionEpsilon = float32(1e-5)
)

// deepSeekOCRTower is one tower of the encoder as the vision config
// declares it under its width table.
type deepSeekOCRTower struct {
	Width              uint32   `json:"width"`
	Layers             uint32   `json:"layers"`
	Heads              uint32   `json:"heads"`
	GlobalAttnIndexes  []uint32 `json:"global_attn_indexes"`
	DownsampleChannels []uint32 `json:"downsample_channels"`
}

// deepSeekOCRProcessor is the preprocessing the checkpoint declares.
type deepSeekOCRProcessor struct {
	ImageMean            []float32  `json:"image_mean"`
	ImageStd             []float32  `json:"image_std"`
	CandidateResolutions [][]uint32 `json:"candidate_resolutions"`
}

// DeepSeekOCRProjectorConversion returns encoder metadata and streamed SAM,
// CLIP and projector tensors for the mmproj GGUF, proved against the
// projector runtime's reader before any byte is written.
func DeepSeekOCRProjectorConversion(repository *hfrepo.Repository) ([]gguf.Metadata, []gguf.TensorData, error) {
	if repository == nil || repository.Tensors == nil {
		return nil, nil, errors.New("HF/GGUF adapter: nil repository")
	}
	if !IsDeepSeekOCRRepository(repository.Identity) {
		return nil, nil, fmt.Errorf("HF/GGUF adapter: decoder architectures %v are not %s", repository.Identity.TextArchitectures, deepSeekOCRLanguageArchitecture)
	}
	mappings, err := collectTensorMappings(repository.Tensors, deepSeekOCRProjectorTensorName, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	metadata, err := deepSeekOCRProjectorMetadata(repository, mappings)
	if err != nil {
		return nil, nil, err
	}
	infos, err := mappedTensorCatalog(mappings)
	if err != nil {
		return nil, nil, err
	}
	catalog, err := gguf.Catalog(metadata, infos)
	if err != nil {
		return nil, nil, err
	}
	if _, err := projector.ReadDeepSeekOCRSpec(catalog); err != nil {
		return nil, nil, fmt.Errorf("HF/GGUF adapter: DeepSeek-OCR projector: %w", err)
	}
	tensors, err := mappedTensorData(mappings)
	if err != nil {
		return nil, nil, err
	}
	return metadata, tensors, nil
}

func deepSeekOCRProjectorMetadata(repository *hfrepo.Repository, mappings []tensorMapping) ([]gguf.Metadata, error) {
	sam, clip, err := deepSeekOCRTowers(repository.Config)
	if err != nil {
		return nil, err
	}
	var processor deepSeekOCRProcessor
	if err := jsonfile.Decode(filepath.Join(repository.Directory, deepSeekOCRProcessorFile), &processor); err != nil {
		return nil, fmt.Errorf("HF/GGUF adapter: DeepSeek-OCR processor config: %w", err)
	}
	if len(processor.ImageMean) != media.RGBChannels || len(processor.ImageStd) != media.RGBChannels || len(processor.CandidateResolutions) == 0 {
		return nil, errors.New("HF/GGUF adapter: DeepSeek-OCR processor config declares no image normalization or resolutions")
	}
	// Every candidate resolution is square, and the tile is that square; the
	// number of candidates bounds the tiles a page may be cut into.
	tile := processor.CandidateResolutions[0]
	for _, resolution := range processor.CandidateResolutions {
		if len(resolution) != 2 || resolution[0] != resolution[1] || resolution[0] != tile[0] {
			return nil, fmt.Errorf("HF/GGUF adapter: DeepSeek-OCR candidate resolutions %v are not one square", processor.CandidateResolutions)
		}
	}
	tiles := uint32(len(processor.CandidateResolutions))
	global := make([]bool, sam.Layers)
	for _, index := range sam.GlobalAttnIndexes {
		if index >= sam.Layers {
			return nil, fmt.Errorf("HF/GGUF adapter: DeepSeek-OCR global attention index %d beyond %d SAM blocks", index, sam.Layers)
		}
		global[index] = true
	}
	window, err := deepSeekOCRWindow(mappings, global)
	if err != nil {
		return nil, err
	}
	return []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "clip"),
		metadata("general.name", gguf.ValueTypeString, filepath.Base(repository.Directory)),
		metadata("clip.projector_type", gguf.ValueTypeString, deepSeekOCRProjectorType),
		metadata("clip.has_vision_encoder", gguf.ValueTypeBool, true),
		metadata("clip.vision.embedding_length", gguf.ValueTypeUint32, clip.Width),
		metadata("clip.vision.block_count", gguf.ValueTypeUint32, clip.Layers),
		metadata("clip.vision.attention.head_count", gguf.ValueTypeUint32, clip.Heads),
		metadata("clip.vision.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, deepSeekOCRDefaultVisionEpsilon),
		metadata("clip.vision.sam.embedding_length", gguf.ValueTypeUint32, sam.Width),
		metadata("clip.vision.sam.block_count", gguf.ValueTypeUint32, sam.Layers),
		metadata("clip.vision.sam.head_count", gguf.ValueTypeUint32, sam.Heads),
		gguf.ArrayMetadata("clip.vision.sam.global_attention_layers", gguf.ValueTypeBool, global),
		metadata("clip.vision.window_size", gguf.ValueTypeUint32, window),
		metadata("clip.vision.preproc_image_size", gguf.ValueTypeUint32, tile[0]),
		metadata("clip.vision.preproc_min_tiles", gguf.ValueTypeUint32, uint32(1)),
		metadata("clip.vision.preproc_max_tiles", gguf.ValueTypeUint32, tiles),
		gguf.ArrayMetadata("clip.vision.image_mean", gguf.ValueTypeFloat32, processor.ImageMean),
		gguf.ArrayMetadata("clip.vision.image_std", gguf.ValueTypeFloat32, processor.ImageStd),
	}, nil
}

// deepSeekOCRTowers reads the two towers from the vision config's width
// table by what they declare: the SAM tower names its global attention
// blocks and the CLIP tower does not.
func deepSeekOCRTowers(config map[string]json.RawMessage) (sam, clip deepSeekOCRTower, err error) {
	var vision struct {
		Width map[string]deepSeekOCRTower `json:"width"`
	}
	if err := json.Unmarshal(config[deepSeekOCRVisionConfig], &vision); err != nil || len(vision.Width) != 2 {
		return sam, clip, errors.New("HF/GGUF adapter: DeepSeek-OCR vision config does not declare two towers")
	}
	samFound, clipFound := false, false
	for _, tower := range vision.Width {
		if tower.Width == 0 || tower.Layers == 0 || tower.Heads == 0 || tower.Width%tower.Heads != 0 {
			return sam, clip, errors.New("HF/GGUF adapter: DeepSeek-OCR tower dimensions are invalid")
		}
		if len(tower.GlobalAttnIndexes) != 0 {
			sam, samFound = tower, true
		} else {
			clip, clipFound = tower, true
		}
	}
	if !samFound || !clipFound {
		return sam, clip, errors.New("HF/GGUF adapter: DeepSeek-OCR vision config needs one SAM and one CLIP tower")
	}
	return sam, clip, nil
}

// deepSeekOCRWindow derives the SAM window from a windowed block's relative
// position table, which spans twice the window less one.
func deepSeekOCRWindow(mappings []tensorMapping, global []bool) (uint32, error) {
	for _, mapping := range mappings {
		rest, ok := strings.CutPrefix(mapping.name, "v.sam.blk.")
		if !ok || !strings.HasSuffix(rest, ".attn.pos_h.weight") {
			continue
		}
		block, _, _ := strings.Cut(rest, ".")
		index, err := strconv.ParseUint(block, binaryschema.DecimalRadix, binaryschema.Width32Bits)
		if err != nil || index >= uint64(len(global)) || global[index] {
			continue
		}
		if len(mapping.shape) != 2 || mapping.shape[0]%2 == 0 {
			return 0, fmt.Errorf("HF/GGUF adapter: DeepSeek-OCR relative position table %s has shape %v", mapping.name, mapping.shape)
		}
		return uint32((mapping.shape[0] + 1) / 2), nil
	}
	return 0, errors.New("HF/GGUF adapter: DeepSeek-OCR SAM tower has no windowed block")
}

// deepSeekOCRSAMNames maps a SAM block tensor suffix to its mmproj suffix.
var deepSeekOCRSAMNames = map[string]string{
	"norm1.weight": "pre_ln.weight", "norm1.bias": "pre_ln.bias",
	"norm2.weight": "post_ln.weight", "norm2.bias": "post_ln.bias",
	"attn.rel_pos_h": "attn.pos_h.weight", "attn.rel_pos_w": "attn.pos_w.weight",
	"attn.qkv.weight": "attn.qkv.weight", "attn.qkv.bias": "attn.qkv.bias",
	"attn.proj.weight": "attn.out.weight", "attn.proj.bias": "attn.out.bias",
	"mlp.lin1.weight": "mlp.lin1.weight", "mlp.lin1.bias": "mlp.lin1.bias",
	"mlp.lin2.weight": "mlp.lin2.weight", "mlp.lin2.bias": "mlp.lin2.bias",
}

// deepSeekOCRCLIPNames maps a CLIP block tensor suffix to its mmproj suffix.
var deepSeekOCRCLIPNames = map[string]string{
	"layer_norm1.weight": "ln1.weight", "layer_norm1.bias": "ln1.bias",
	"layer_norm2.weight": "ln2.weight", "layer_norm2.bias": "ln2.bias",
	"self_attn.qkv_proj.weight": "attn_qkv.weight", "self_attn.qkv_proj.bias": "attn_qkv.bias",
	"self_attn.out_proj.weight": "attn_out.weight", "self_attn.out_proj.bias": "attn_out.bias",
	"mlp.fc1.weight": "ffn_up.weight", "mlp.fc1.bias": "ffn_up.bias",
	"mlp.fc2.weight": "ffn_down.weight", "mlp.fc2.bias": "ffn_down.bias",
}

// deepSeekOCRProjectorTensorName maps an encoder or projector tensor to its
// mmproj name; decoder tensors are the language conversion's.
func deepSeekOCRProjectorTensorName(name string) (string, bool, error) {
	switch name {
	case "model.sam_model.pos_embed":
		return "v.sam.pos_embd.weight", true, nil
	case "model.sam_model.patch_embed.proj.weight":
		return "v.sam.patch_embd.weight", true, nil
	case "model.sam_model.patch_embed.proj.bias":
		return "v.sam.patch_embd.bias", true, nil
	case "model.sam_model.net_2.weight", "model.sam_model.net_3.weight":
		return "v.sam." + strings.TrimPrefix(name, "model.sam_model."), true, nil
	case "model.vision_model.embeddings.class_embedding":
		return "v.class_embd", true, nil
	case "model.vision_model.embeddings.position_embedding.weight":
		return "v.position_embd.weight", true, nil
	case "model.vision_model.embeddings.patch_embedding.weight":
		return "v.patch_embd.weight", true, nil
	case "model.vision_model.pre_layrnorm.weight":
		return "v.pre_ln.weight", true, nil
	case "model.vision_model.pre_layrnorm.bias":
		return "v.pre_ln.bias", true, nil
	case "model.vision_model.post_layernorm.weight":
		return "v.post_ln.weight", true, nil
	case "model.vision_model.post_layernorm.bias":
		return "v.post_ln.bias", true, nil
	case "model.projector.layers.weight":
		return "mm.model.fc.weight", true, nil
	case "model.projector.layers.bias":
		return "mm.model.fc.bias", true, nil
	case "model.image_newline":
		return "v.image_newline", true, nil
	case "model.view_seperator":
		return "v.view_seperator", true, nil
	}
	if rest, ok := strings.CutPrefix(name, "model.sam_model.neck."); ok {
		return "v.sam.neck." + rest, true, nil
	}
	if strings.HasPrefix(name, "model.sam_model.blocks.") {
		mapped, _, err := mapLayerTensor(name, "model.sam_model.blocks.", 0, deepSeekOCRSAMNames)
		if err != nil {
			return "", false, fmt.Errorf("HF/GGUF adapter: SAM tensor %q has no mapping", name)
		}
		return "v.sam." + mapped, true, nil
	}
	if strings.HasPrefix(name, "model.vision_model.transformer.layers.") {
		mapped, _, err := mapLayerTensor(name, "model.vision_model.transformer.layers.", 0, deepSeekOCRCLIPNames)
		if err != nil {
			return "", false, fmt.Errorf("HF/GGUF adapter: CLIP tensor %q has no mapping", name)
		}
		return "v." + mapped, true, nil
	}
	return "", false, nil // the decoder
}
