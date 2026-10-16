package trainingworkflow

import (
	"fmt"
	"maps"
	"math"
	"os"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"path/filepath"
	"slices"
	"testing"
)

// Tiny speech geometry: model and flow width, attention heads, feed-forward
// width, text vocabulary, and latent width; every depth is one.
const (
	tinySpeechWidth, tinySpeechHeads, tinySpeechFF = 4, 2, 8
	tinySpeechVocab, tinySpeechLatent              = 6, 2
)

// writeTinySpeech writes a synthetic speech synthesis checkpoint, one
// backbone layer, one flow block, a one-layer codec transformer and a
// one-stage SEANet, with its SentencePiece tokenizer, as a model directory
// the speechsynth loader reads, and returns the directory.
func writeTinySpeech(t *testing.T, directory string) string {
	t.Helper()
	d, ff, latent, fused := tinySpeechWidth, tinySpeechFF, tinySpeechLatent, 3*tinySpeechWidth
	shapes := map[string][]int{
		"flow_lm.conditioner.embed.weight": {tinySpeechVocab, d}, "flow_lm.input_linear.weight": {d, latent},
		"flow_lm.out_norm.weight": {d}, "flow_lm.out_norm.bias": {d}, "flow_lm.out_eos.weight": {1, d}, "flow_lm.out_eos.bias": {1},
		"flow_lm.bos_emb": {latent}, "flow_lm.emb_mean": {latent}, "flow_lm.emb_std": {latent},
		"flow_lm.flow_net.input_proj.weight": {d, latent}, "flow_lm.flow_net.input_proj.bias": {d},
		"flow_lm.flow_net.cond_embed.weight": {d, d}, "flow_lm.flow_net.cond_embed.bias": {d},
		"flow_lm.flow_net.final_layer.adaLN_modulation.1.weight": {2 * d, d}, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias": {2 * d},
		"flow_lm.flow_net.final_layer.linear.weight": {latent, d}, "flow_lm.flow_net.final_layer.linear.bias": {latent},
		"mimi.quantizer.output_proj.weight": {d, latent, 1}, "mimi.upsample.convtr.convtr.weight": {d, 1, 2},
		"mimi.decoder.model.0.conv.weight": {2, d, 3}, "mimi.decoder.model.0.conv.bias": {2},
		"mimi.decoder.model.2.convtr.weight": {2, 1, 4}, "mimi.decoder.model.2.convtr.bias": {1},
		"mimi.decoder.model.3.block.1.conv.weight": {1, 1, 3}, "mimi.decoder.model.3.block.1.conv.bias": {1},
		"mimi.decoder.model.3.block.3.conv.weight": {1, 1, 1}, "mimi.decoder.model.3.block.3.conv.bias": {1},
		"mimi.decoder.model.5.conv.weight": {1, 1, 3}, "mimi.decoder.model.5.conv.bias": {1},
	}
	transformer := func(prefix string) {
		for name, shape := range map[string][]int{
			"norm1.weight": {d}, "norm1.bias": {d}, "norm2.weight": {d}, "norm2.bias": {d},
			"self_attn.in_proj.weight": {fused, d}, "self_attn.out_proj.weight": {d, d},
			"linear1.weight": {ff, d}, "linear2.weight": {d, ff},
		} {
			shapes[prefix+name] = shape
		}
	}
	transformer("flow_lm.transformer.layers.0.")
	transformer("mimi.decoder_transformer.transformer.layers.0.")
	shapes["mimi.decoder_transformer.transformer.layers.0.layer_scale_1.scale"] = []int{d}
	shapes["mimi.decoder_transformer.transformer.layers.0.layer_scale_2.scale"] = []int{d}
	for embedder := range 2 {
		prefix := fmt.Sprintf("flow_lm.flow_net.time_embed.%d.", embedder)
		for name, shape := range map[string][]int{
			"freqs": {2}, "mlp.0.weight": {d, 4}, "mlp.0.bias": {d}, "mlp.2.weight": {d, d}, "mlp.2.bias": {d}, "mlp.3.alpha": {d},
		} {
			shapes[prefix+name] = shape
		}
	}
	for name, shape := range map[string][]int{
		"in_ln.weight": {d}, "in_ln.bias": {d}, "mlp.0.weight": {d, d}, "mlp.0.bias": {d}, "mlp.2.weight": {d, d}, "mlp.2.bias": {d},
		"adaLN_modulation.1.weight": {3 * d, d}, "adaLN_modulation.1.bias": {3 * d},
	} {
		shapes["flow_lm.flow_net.res_blocks.0."+name] = shape
	}
	tensors := make(map[string][]float32, len(shapes))
	for index, name := range slices.Sorted(maps.Keys(shapes)) {
		values := make([]float32, speechElements(shapes[name]))
		for element := range values {
			values[element] = float32(math.Sin(float64(index*len(values)+element+1))) / 2
		}
		tensors[name] = values
	}
	// The latent statistics are the identity normalization.
	for element := range latent {
		tensors["flow_lm.emb_mean"][element], tensors["flow_lm.emb_std"][element] = 0, 1
	}
	config := `{"default_temperature":1,"flow_lm":{"transformer":{"max_period":10000,"num_heads":2}},"mimi":{"frame_rate":12.5,"sample_rate":24000,` +
		`"seanet":{"channels":1,"compress":1,"dimension":4,"kernel_size":3,"last_kernel_size":3,"n_filters":1,"ratios":[2],"residual_kernel_size":3},` +
		`"transformer":{"context":4,"max_period":10000,"num_heads":2}}}`
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "pockettts_config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), tensors, shapes, nil); err != nil {
		t.Fatal(err)
	}
	testutil.WriteSentencePiece(t, filepath.Join(directory, "tokenizer.model"),
		[]string{"<pad>", "</s>", "<unk>", "▁a", "▁b", "▁c"}, 2)
	return directory
}

func speechElements(shape []int) int {
	result := 1
	for _, n := range shape {
		result *= n
	}
	return result
}
