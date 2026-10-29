package hfgguf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/safetensors"
)

// TestQwen35ZeroCenteredNormsShiftByOne: every zero-centered RMSNorm the conversion writes is stored as 1+w, the draft
// layer's input norms with the rest (a Qwen3.5-4B converted without them drafted almost no token its trunk kept), and the
// gated-delta group norm stays raw.
func TestQwen35ZeroCenteredNormsShiftByOne(t *testing.T) {
	t.Parallel()
	transforms := qwen35TensorTransforms(qwen35LinearDims{keyHeads: 1, valueHeads: 1, keyWidth: 1, valueWidth: 1})
	for name, want := range map[string]float32{
		"model.language_model.layers.0.input_layernorm.weight":  1.5,
		"model.language_model.layers.0.self_attn.q_norm.weight": 1.5,
		"model.language_model.norm.weight":                      1.5,
		"mtp.pre_fc_norm_embedding.weight":                      1.5,
		"mtp.pre_fc_norm_hidden.weight":                         1.5,
		"mtp.norm.weight":                                       1.5,
		"model.language_model.layers.0.linear_attn.norm.weight": 0.5,
	} {
		payload := binary.LittleEndian.AppendUint32(nil, math.Float32bits(0.5))
		if transform := transforms(name); transform != nil {
			if err := transform(payload, 4); err != nil {
				t.Fatal(err)
			}
		}
		if got := math.Float32frombits(binary.LittleEndian.Uint32(payload)); got != want {
			t.Errorf("%s: stored %v from 0.5, want %v", name, got, want)
		}
	}
}

func TestQwen35RecurrentMetadataIncludesPredictionLayers(t *testing.T) {
	for name, schedule := range map[string][]string{
		"periodic": {"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		"explicit": {"full_attention", "linear_attention", "full_attention", "linear_attention"},
	} {
		t.Run(name, func(t *testing.T) {
			repository := qwen35Fixture(t)
			var text map[string]json.RawMessage
			if err := json.Unmarshal(repository.Config["text_config"], &text); err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(schedule)
			if err != nil {
				t.Fatal(err)
			}
			text["layer_types"] = encoded
			repository.Config["text_config"], err = json.Marshal(text)
			if err != nil {
				t.Fatal(err)
			}
			metadata, trunk, err := qwen35Metadata(repository)
			if err != nil {
				t.Fatal(err)
			}
			var declared, prediction uint32
			var recurrent []bool
			for _, item := range metadata {
				switch item.Key {
				case "qwen35.block_count":
					declared = item.Value.Data.(uint32)
				case "qwen35.nextn_predict_layers":
					prediction = item.Value.Data.(uint32)
				case "qwen35.attention.recurrent_layers":
					recurrent = item.Value.Data.([]bool)
				}
			}
			if trunk != uint32(len(schedule)) || declared != trunk+prediction || prediction != 1 || len(recurrent) != int(declared) {
				t.Fatalf("native layer extent: trunk=%d prediction=%d declared=%d recurrent=%d", trunk, prediction, declared, len(recurrent))
			}
			for index, layer := range schedule {
				if recurrent[index] != (layer == "linear_attention") {
					t.Fatalf("trunk layer %d changed its declared attention type", index)
				}
			}
			if slices.Contains(recurrent[trunk:], true) {
				t.Fatal("prediction layer must use dense attention")
			}
		})
	}
}

func TestQwen35ConvolutionNativeStorage(t *testing.T) {
	repository := qwen35Fixture(t)
	const name = "model.language_model.layers.0.linear_attn.conv1d.weight"
	tensor := repository.Tensors.Tensors[name]
	encoded := make([]byte, tensor.Elements()*2)
	for offset := 0; offset < len(encoded); offset += 2 {
		binary.LittleEndian.PutUint16(encoded[offset:], uint16(math.Float32bits(1)>>16))
	}
	var err error
	repository.Tensors.Tensors[name], err = safetensors.NewTensor(name, "BF16", tensor.Shape, bytes.NewReader(encoded), 0, int64(len(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	_, tensors, err := Qwen35Conversion(repository)
	if err != nil {
		t.Fatal(err)
	}
	for _, output := range tensors {
		if output.Name != "blk.0.ssm_conv1d.weight" {
			continue
		}
		if output.Type != gguf.DTypeF32 {
			t.Fatalf("native convolution requires F32, got %s", output.Type)
		}
		data, err := io.ReadAll(output.Data)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != int(tensor.Elements())*4 {
			t.Fatal("widening changed the element denominator")
		}
		for offset := 0; offset < len(data); offset += 4 {
			if binary.LittleEndian.Uint32(data[offset:]) != math.Float32bits(1) {
				t.Fatal("widening changed a convolution value")
			}
		}
		return
	}
	t.Fatal("converted convolution absent")
}
