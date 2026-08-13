package gemma4convert

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/projector"
)

func TestTowerTensorName(t *testing.T) {
	for source, want := range map[string]string{
		"model.vision_tower.patch_embedder.input_proj.weight":                    "v.patch_embd.weight",
		"model.vision_tower.patch_embedder.position_embedding_table":             "v.position_embd.weight",
		"model.vision_tower.encoder.layers.3.input_layernorm.weight":             "v.blk.3.attn_norm.weight",
		"model.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight":     "v.blk.3.attn_q.weight",
		"model.vision_tower.encoder.layers.3.self_attn.q_proj.input_min":         "v.blk.3.attn_q.input_min",
		"model.vision_tower.encoder.layers.3.self_attn.k_norm.weight":            "v.blk.3.attn_k_norm.weight",
		"model.vision_tower.encoder.layers.15.mlp.down_proj.output_max":          "v.blk.15.ffn_down.output_max",
		"model.vision_tower.encoder.layers.15.post_feedforward_layernorm.weight": "v.blk.15.post_ffw_norm.weight",
		"model.embed_vision.embedding_projection.weight":                         "mm.input_projection.weight",
		"model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight":     "a.blk.0.ffn1_up.weight",
		"model.audio_tower.layers.0.feed_forward2.ffw_layer_2.input_max":         "a.blk.0.ffn2_down.input_max",
		"model.audio_tower.layers.11.lconv1d.depthwise_conv1d.weight":            "a.blk.11.conv_dw.weight",
		"model.audio_tower.layers.11.lconv1d.linear_start.output_min":            "a.blk.11.conv_start.output_min",
		"model.audio_tower.layers.11.self_attn.per_dim_scale":                    "a.blk.11.attn_per_dim_scale.weight",
		"model.audio_tower.layers.11.self_attn.relative_k_proj.weight":           "a.blk.11.attn_rel_k.weight",
		"model.audio_tower.layers.11.self_attn.post.linear.weight":               "a.blk.11.attn_output.weight",
		"model.audio_tower.layers.11.norm_out.weight":                            "a.blk.11.out_norm.weight",
		"model.audio_tower.subsample_conv_projection.layer0.conv.weight":         "a.conv.0.weight",
		"model.audio_tower.subsample_conv_projection.layer1.norm.weight":         "a.conv.1.norm.weight",
		"model.audio_tower.subsample_conv_projection.input_proj_linear.weight":   "a.input_proj.weight",
		"model.audio_tower.output_proj.bias":                                     "a.output_proj.bias",
		"model.embed_audio.embedding_projection.weight":                          "mm.a.input_projection.weight",
	} {
		got, ok := towerTensorName(source)
		if !ok || got != want {
			t.Fatalf("mapping %q = %q, %t; want %q", source, got, ok, want)
		}
	}
	for _, unmapped := range []string{
		"model.language_model.layers.0.self_attn.q_proj.weight",
		"model.vision_tower.encoder.layers.3.self_attn.q_proj.weight",
		"model.audio_tower.layers.0.unknown.weight",
	} {
		if got, ok := towerTensorName(unmapped); ok {
			t.Fatalf("unexpected mapping %q = %q", unmapped, got)
		}
	}
}

func towerTestConfig() (modelConfig, processorConfig) {
	var config modelConfig
	config.Text.HiddenSize = 16
	vision := &config.Vision
	vision.HiddenSize, vision.HiddenLayers, vision.AttentionHeads = 8, 2, 2
	vision.KVHeads, vision.HeadDim, vision.IntermediateSize = 2, 4, 12
	vision.PatchSize, vision.PoolingSize, vision.PositionEmbedding = 2, 3, 5
	vision.RMSEpsilon, vision.Rope.Theta, vision.HiddenAct = 1e-6, 100, "gelu_pytorch_tanh"
	audio := &config.Audio
	audio.HiddenSize, audio.HiddenLayers, audio.AttentionHeads = 8, 2, 2
	audio.ConvKernel, audio.SubChannels, audio.ChunkSize = 3, []uint32{4, 2}, 2
	audio.ContextLeft, audio.ContextRight = 1, 0
	audio.LogitCap, audio.ResidualWeight, audio.RMSEpsilon = 50, 0.5, 1e-6
	audio.OutputProjDims, audio.HiddenAct = 6, "silu"
	var processor processorConfig
	processor.Audio.FeatureSize, processor.Audio.FFTLength = 8, 16
	processor.Audio.FrameLength, processor.Audio.HopLength, processor.Audio.SampleRate = 4, 2, 16000
	processor.Audio.MinFrequency, processor.Audio.MaxFrequency, processor.Audio.MelFloor = 0, 8000, 0.001
	processor.Image.Mean, processor.Image.Std = []float32{0, 0, 0}, []float32{1, 1, 1}
	processor.Image.MaxSoftTokens, processor.Video.MaxSoftTokens = 7, 3
	return config, processor
}

// towerTestSourceShapes: independent enumeration of the HF checkpoint
// structure for a given config; shapes in source (row-major) order.
func towerTestSourceShapes(config modelConfig) map[string][]uint64 {
	vision, audio := config.Vision, config.Audio
	text := uint64(config.Text.HiddenSize)
	vh := uint64(vision.HiddenSize)
	pixels := uint64(vision.PatchSize * vision.PatchSize * rgbChannelCount)
	qw := uint64(vision.AttentionHeads * vision.HeadDim)
	kw := uint64(vision.KVHeads * vision.HeadDim)
	vi := uint64(vision.IntermediateSize)
	ah := uint64(audio.HiddenSize)
	const audioInter = uint64(24)
	shapes := map[string][]uint64{
		"model.vision_tower.patch_embedder.input_proj.weight":                  {vh, pixels},
		"model.vision_tower.patch_embedder.position_embedding_table":           {2, uint64(vision.PositionEmbedding), vh},
		"model.embed_vision.embedding_projection.weight":                       {text, vh},
		"model.audio_tower.subsample_conv_projection.input_proj_linear.weight": {ah, uint64(audio.SubChannels[1]) * uint64(processorTestMelQuarter)},
		"model.audio_tower.output_proj.weight":                                 {uint64(audio.OutputProjDims), ah},
		"model.audio_tower.output_proj.bias":                                   {uint64(audio.OutputProjDims)},
		"model.embed_audio.embedding_projection.weight":                        {text, uint64(audio.OutputProjDims)},
	}
	inputChannels := uint64(1)
	for index, channels := range audio.SubChannels {
		prefix := fmt.Sprintf("model.audio_tower.subsample_conv_projection.layer%d.", index)
		shapes[prefix+"conv.weight"] = []uint64{uint64(channels), inputChannels, 3, 3}
		shapes[prefix+"norm.weight"] = []uint64{uint64(channels)}
		inputChannels = uint64(channels)
	}
	clipped := func(prefix string, shape []uint64) {
		shapes[prefix+".linear.weight"] = shape
		for _, scalar := range []string{"input_min", "input_max", "output_min", "output_max"} {
			shapes[prefix+"."+scalar] = nil
		}
	}
	for layer := uint32(0); layer < vision.HiddenLayers; layer++ {
		prefix := fmt.Sprintf("model.vision_tower.encoder.layers.%d.", layer)
		for _, norm := range []string{
			"input_layernorm", "post_attention_layernorm",
			"pre_feedforward_layernorm", "post_feedforward_layernorm",
		} {
			shapes[prefix+norm+".weight"] = []uint64{vh}
		}
		shapes[prefix+"self_attn.q_norm.weight"] = []uint64{uint64(vision.HeadDim)}
		shapes[prefix+"self_attn.k_norm.weight"] = []uint64{uint64(vision.HeadDim)}
		clipped(prefix+"self_attn.q_proj", []uint64{qw, vh})
		clipped(prefix+"self_attn.k_proj", []uint64{kw, vh})
		clipped(prefix+"self_attn.v_proj", []uint64{kw, vh})
		clipped(prefix+"self_attn.o_proj", []uint64{vh, qw})
		clipped(prefix+"mlp.gate_proj", []uint64{vi, vh})
		clipped(prefix+"mlp.up_proj", []uint64{vi, vh})
		clipped(prefix+"mlp.down_proj", []uint64{vh, vi})
	}
	for layer := uint32(0); layer < audio.HiddenLayers; layer++ {
		prefix := fmt.Sprintf("model.audio_tower.layers.%d.", layer)
		for _, norm := range []string{"norm_pre_attn", "norm_post_attn", "norm_out"} {
			shapes[prefix+norm+".weight"] = []uint64{ah}
		}
		for _, ffw := range []string{"feed_forward1", "feed_forward2"} {
			shapes[prefix+ffw+".pre_layer_norm.weight"] = []uint64{ah}
			shapes[prefix+ffw+".post_layer_norm.weight"] = []uint64{ah}
			clipped(prefix+ffw+".ffw_layer_1", []uint64{audioInter, ah})
			clipped(prefix+ffw+".ffw_layer_2", []uint64{ah, audioInter})
		}
		shapes[prefix+"lconv1d.pre_layer_norm.weight"] = []uint64{ah}
		shapes[prefix+"lconv1d.conv_norm.weight"] = []uint64{ah}
		shapes[prefix+"lconv1d.depthwise_conv1d.weight"] = []uint64{ah, 1, uint64(audio.ConvKernel)}
		clipped(prefix+"lconv1d.linear_start", []uint64{2 * ah, ah})
		clipped(prefix+"lconv1d.linear_end", []uint64{ah, ah})
		shapes[prefix+"self_attn.per_dim_scale"] = []uint64{ah / uint64(audio.AttentionHeads)}
		shapes[prefix+"self_attn.relative_k_proj.weight"] = []uint64{ah, ah}
		clipped(prefix+"self_attn.q_proj", []uint64{ah, ah})
		clipped(prefix+"self_attn.k_proj", []uint64{ah, ah})
		clipped(prefix+"self_attn.v_proj", []uint64{ah, ah})
		clipped(prefix+"self_attn.post", []uint64{ah, ah})
	}
	return shapes
}

const processorTestMelQuarter = 2 // melBins/4 after two stride-2 subsample convs

func TestTowerConvertMatchesPinnedLoader(t *testing.T) {
	config, processor := towerTestConfig()
	if !towerLayout(config) {
		t.Fatal("tower layout not detected")
	}
	if err := validateTowerConfig(config, processor); err != nil {
		t.Fatal(err)
	}
	const audioInter = 24
	metadata := towerProjectorMetadata("fixture", config, processor, audioInter)
	shapes := towerTestSourceShapes(config)
	tensors := make([]gguf.TensorData, 0, len(shapes))
	for source, shape := range shapes {
		name, ok := towerTensorName(source)
		if !ok {
			t.Fatalf("source %q has no mapping", source)
		}
		destinationShape := reverseShape(shape)
		dataType := gguf.DTypeBF16
		elementBytes := uint64(bf16StorageBytes)
		if len(destinationShape) == 0 {
			destinationShape = []uint64{1}
			dataType = gguf.DTypeF32
			elementBytes = float32StorageBytes
		}
		size := elementBytes
		for _, dimension := range destinationShape {
			size *= dimension
		}
		tensors = append(tensors, gguf.TensorData{
			Name: name, Shape: destinationShape, Type: dataType,
			Data: bytes.NewReader(make([]byte, size)),
		})
	}
	path := filepath.Join(t.TempDir(), "tower.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	runner, err := projector.OpenGemma4Tower(path)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	spec := runner.Spec()
	if spec.Vision.Layers != int(config.Vision.HiddenLayers) ||
		spec.Vision.Hidden != int(config.Vision.HiddenSize) ||
		spec.Vision.ProjectionDim != int(config.Text.HiddenSize) ||
		spec.Audio.Layers != int(config.Audio.HiddenLayers) ||
		spec.Audio.Intermediate != audioInter ||
		spec.Audio.MelBins != int(processor.Audio.FeatureSize) ||
		len(spec.Audio.SubChannels) != len(config.Audio.SubChannels) {
		t.Fatalf("spec = %+v", spec)
	}
	output, err := runner.EncodeVisionImage(context.Background(), image.NewRGBA(image.Rect(0, 0, 6, 6)))
	if err != nil {
		t.Fatal(err)
	}
	if output.PatchCount != 9 || output.SoftTokens != 1 ||
		output.Embeddings.Shape.Dims[0] != uint64(config.Text.HiddenSize) || len(output.Embeddings.Data) != int(config.Text.HiddenSize) {
		t.Fatalf("tower vision output = %+v", output)
	}
	for index, value := range output.Embeddings.Data {
		if value != 0 {
			t.Fatalf("tower vision output[%d] = %v, want zero", index, value)
		}
	}
}
