package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingprogram"
)

// TestSeq2SeqRoutePublishesCheckpoint trains a tiny synthetic
// encoder-decoder checkpoint through the seq2seq route on a JSON Lines
// dataset of text pairs and holds the route to publishing the trained model
// as a checkpoint recorded in the store, one tensor per trained parameter.
func TestSeq2SeqRoutePublishesCheckpoint(t *testing.T) {
	if runtime.GOOS == "windows" {
		cudatest.Require(t)
	}
	root := t.TempDir()
	model := writeTinySeq2Seq(t, filepath.Join(root, "model"))
	dataset := filepath.Join(root, "pairs.jsonl")
	if err := os.WriteFile(dataset, []byte("{\"question\":\"a b\",\"answer\":\"c\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	checkpoint, out := requireRoutePublishes(t, "seq2seq", "-model", model, "-dataset", dataset, "-input-field", "question", "-target-field", "answer")
	weights, err := safetensors.OpenFile(filepath.Join(out, trainingprogram.CheckpointWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	var elements uint64
	for _, tensor := range weights.Tensors {
		elements += tensor.Elements()
	}
	if _, trained := weights.Tensors["shared.embedding"]; !trained || elements != uint64(checkpoint.ParameterCount) {
		t.Fatalf("checkpoint tensors hold %d values, want the %d trained parameters", elements, checkpoint.ParameterCount)
	}
}

// Tiny encoder-decoder geometry: vocabulary, width, query and key-value
// heads, head width, and encoder and decoder depth.
const (
	tinySeq2SeqVocab, tinySeq2SeqWidth, tinySeq2SeqHeads = 8, 4, 2
	tinySeq2SeqKVHeads, tinySeq2SeqHeadDim               = 1, 2
	tinySeq2SeqEncoderLayers, tinySeq2SeqDecoderLayers   = 1, 2
)

// writeTinySeq2Seq writes a synthetic encoder-decoder checkpoint in the
// transformers layout the seq2seq loader reads, with its SentencePiece
// tokenizer, and returns its directory.
func writeTinySeq2Seq(t *testing.T, directory string) string {
	t.Helper()
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	config := `{"model_type":"needle","rms_norm_eps":1e-6,"rope_theta":10000,"decoder_start_token_id":1,"eos_token_id":1}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	d, qWidth, kvWidth := tinySeq2SeqWidth, tinySeq2SeqHeads*tinySeq2SeqHeadDim, tinySeq2SeqKVHeads*tinySeq2SeqHeadDim
	shapes := map[string][]int{
		"model.embed_tokens.weight":       {tinySeq2SeqVocab, d},
		"model.encoder.final_norm.weight": {d},
		"model.decoder.norm.weight":       {d},
	}
	attention := func(prefix, norm, gate string) {
		shapes[prefix+norm] = []int{d}
		shapes[prefix+gate] = []int{1}
	}
	projections := func(prefix string) {
		for name, shape := range map[string][]int{
			"q_proj.weight": {qWidth, d}, "k_proj.weight": {kvWidth, d}, "v_proj.weight": {kvWidth, d},
			"out_proj.weight": {d, qWidth}, "q_norm.weight": {tinySeq2SeqHeadDim}, "k_norm.weight": {tinySeq2SeqHeadDim},
		} {
			shapes[prefix+name] = shape
		}
	}
	for layer := range tinySeq2SeqEncoderLayers {
		prefix := fmt.Sprintf("model.encoder.layers.%d.", layer)
		attention(prefix, "input_layernorm.weight", "attn_gate")
		projections(prefix + "self_attn.")
	}
	for layer := range tinySeq2SeqDecoderLayers {
		prefix := fmt.Sprintf("model.decoder.layers.%d.", layer)
		attention(prefix, "input_layernorm.weight", "self_attn_gate")
		attention(prefix, "encoder_attn_layer_norm.weight", "cross_attn_gate")
		projections(prefix + "self_attn.")
		projections(prefix + "encoder_attn.")
	}
	tensors := make(map[string][]float32, len(shapes))
	for index, name := range slices.Sorted(maps.Keys(shapes)) {
		values := make([]float32, elementCount(shapes[name]))
		for element := range values {
			values[element] = float32(math.Sin(float64(index*len(values)+element+1))) / 2
		}
		tensors[name] = values
	}
	writeBF16Safetensors(t, filepath.Join(directory, trainingprogram.CheckpointWeights), tensors, shapes)
	writeSentencePiece(t, filepath.Join(directory, "tokenizer.model"),
		[]string{"<pad>", "</s>", "<unk>", "▁a", "▁b", "▁c", "▁d", "▁e"}, 2)
	return directory
}

// writeBF16Safetensors writes tensors, rounded to BF16, as one safetensors
// file.
func writeBF16Safetensors(t *testing.T, path string, tensors map[string][]float32, shapes map[string][]int) {
	t.Helper()
	header := map[string]any{}
	var data []byte
	for _, name := range slices.Sorted(maps.Keys(tensors)) {
		begin := len(data)
		for _, value := range tensors[name] {
			data = binary.LittleEndian.AppendUint16(data, dtype.Float32ToBF16(value))
		}
		header[name] = map[string]any{"dtype": "BF16", "shape": shapes[name], "data_offsets": []int{begin, len(data)}}
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(encoded)))
	if err := os.WriteFile(path, append(append(file, encoded...), data...), 0o600); err != nil {
		t.Fatal(err)
	}
}

// writeSentencePiece writes a SentencePiece ModelProto holding only the
// piece table: every piece normal, unknown typed UNKNOWN.
func writeSentencePiece(t *testing.T, path string, pieces []string, unknown int) {
	t.Helper()
	const (
		pieceEntry, pieceText, pieceType = 0x0a, 0x0a, 0x18
		normal, unknownType              = 1, 2
	)
	var model []byte
	for index, piece := range pieces {
		entry := binary.AppendUvarint([]byte{pieceText}, uint64(len(piece)))
		entry = append(entry, piece...)
		kind := uint64(normal)
		if index == unknown {
			kind = unknownType
		}
		entry = binary.AppendUvarint(append(entry, pieceType), kind)
		model = binary.AppendUvarint(append(model, pieceEntry), uint64(len(entry)))
		model = append(model, entry...)
	}
	if err := os.WriteFile(path, model, 0o600); err != nil {
		t.Fatal(err)
	}
}
