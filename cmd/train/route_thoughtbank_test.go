package main

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/pytorchzip"
	"overgo/internal/safetensors"
	"overgo/internal/thoughtbank"
	"overgo/internal/trainingprogram"
)

// TestThoughtBankRoutePublishesCheckpoint trains the first block's bank of
// a tiny synthetic thoughtbank checkpoint through the thoughtbank route and
// holds the route to publishing the trained bank, and only the bank, as a
// checkpoint recorded in the store.
func TestThoughtBankRoutePublishesCheckpoint(t *testing.T) {
	root := t.TempDir()
	model := writeTinyThoughtBank(t, filepath.Join(root, "model"))
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, []byte("abcdabcd"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, out := requireRoutePublishes(t, "thoughtbank", "-model", model, "-dataset", dataset, "-rows", "3")
	weights, err := safetensors.OpenFile(filepath.Join(out, trainingprogram.CheckpointWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	bank := []string{"bank.fw_a", "bank.fw_b", "bank.fw_o", "bank.norm"}
	for _, name := range bank {
		if _, trained := weights.Tensors[name]; !trained {
			t.Errorf("checkpoint lacks the trained bank tensor %s", name)
		}
	}
	if len(weights.Tensors) != len(bank) {
		t.Errorf("checkpoint holds %d tensors, want the bank's %d", len(weights.Tensors), len(bank))
	}
}

// Tiny thoughtbank geometry: vocabulary and model width, head width, query
// latent width, expert width, and memory width; every other count is one.
const (
	tinyBankVocab, tinyBankWidth, tinyBankHeadDim = 4, 4, 2
	tinyBankLatentQ, tinyBankFF, tinyBankMemory   = 2, 2, 2
	tinyBankLayers                                = 2
)

// writeTinyThoughtBank writes a synthetic thoughtbank checkpoint, one
// sparse and one dense block, and its byte tokenizer over "abcd" as a model
// directory the thoughtbank route loads, and returns the directory. Tensors
// no shape derivation or validation reads hold one value.
func writeTinyThoughtBank(t *testing.T, directory string) string {
	t.Helper()
	d := tinyBankWidth
	shapes := map[string][]int64{}
	for _, name := range thoughtbank.ExpectedTensorNames(thoughtbank.ArchConfig{NLayers: tinyBankLayers, NGroups: 1, NExperts: 1, NShared: 1}) {
		shapes[name] = []int64{1}
	}
	shapes["embed.weight"] = []int64{tinyBankVocab, int64(d)}
	shapes["blocks.0._attn.W_uq.weight"] = []int64{tinyBankHeadDim, tinyBankLatentQ}
	shapes["blocks.0._attn.W_dq.weight"] = []int64{tinyBankLatentQ, int64(d)}
	shapes["blocks.0._attn.pos_a"] = []int64{1, tinyBankHeadDim}
	shapes["blocks.1._attn.pos"] = []int64{1, tinyBankHeadDim}
	shapes["blocks.0._moe.experts.0.w3.weight"] = []int64{int64(d), tinyBankFF}
	for layer := range tinyBankLayers {
		prefix := "blocks." + strconv.Itoa(layer) + "."
		shapes[prefix+"_attn.q_norm.weight"] = []int64{tinyBankHeadDim}
		for name, shape := range map[string][]int64{
			"fw_A.weight": {int64(d), tinyBankMemory}, "fw_B.weight": {int64(d), tinyBankMemory},
			"fw_o.weight": {int64(d), int64(d)}, "norm_fw.weight": {int64(d)},
		} {
			shapes[prefix+name] = shape
		}
		for _, connection := range []string{"mhc_attn", "mhc_moe"} {
			for _, name := range []string{".W_pre.weight", ".W_res.weight", ".W_post.weight", ".norm.weight"} {
				shapes[prefix+connection+name] = []int64{int64(d)}
			}
		}
	}
	var pickle bytes.Buffer
	pickle.Write([]byte{0x80, 2, '}', 'q', 0, '('}) // PROTO 2, an empty dict, MARK
	var archive bytes.Buffer
	files := zip.NewWriter(&archive)
	var index int
	for name, shape := range shapes {
		key := strconv.Itoa(index)
		index++
		count, stride := int64(1), make([]int64, len(shape))
		for axis := len(shape) - 1; axis >= 0; axis-- {
			stride[axis] = count
			count *= shape[axis]
		}
		writePickleTensor(&pickle, name, key, count, shape, stride)
		storage, err := files.CreateHeader(&zip.FileHeader{Name: "archive/data/" + key, Method: zip.Store})
		if err != nil {
			t.Fatal(err)
		}
		for element := range count {
			value := float32(math.Sin(float64(int64(index)*count+element))) / 2
			if err := binary.Write(storage, binary.LittleEndian, value); err != nil {
				t.Fatal(err)
			}
		}
	}
	writePickleString(&pickle, "mem_read_swiglu")
	pickle.WriteByte(0x89) // NEWFALSE
	for _, name := range []string{"max_mem", "top_k_csa", "top_k_experts", "n_win", "sinkhorn_iters", "mem_seed_slots"} {
		writePickleString(&pickle, name)
		writePickleInt(&pickle, 1)
	}
	pickle.Write([]byte{'u', '.'}) // SETITEMS, STOP
	data, err := files.CreateHeader(&zip.FileHeader{Name: "archive/data.pkl", Method: zip.Deflate})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := data.Write(pickle.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := files.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, thoughtbankCheckpoint), archive.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	tokenizer := `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}`
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(tokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
	return directory
}

// writePickleTensor appends one state_dict entry: name, then the
// torch._utils._rebuild_tensor_v2 reduction over a float32 storage.
func writePickleTensor(pickle *bytes.Buffer, name, key string, count int64, shape, stride []int64) {
	writePickleString(pickle, name)
	writePickleGlobal(pickle, "torch._utils", "_rebuild_tensor_v2")
	pickle.WriteString("((") // the arguments, then the persistent storage id
	writePickleString(pickle, "storage")
	writePickleGlobal(pickle, "torch", pytorchzip.Float32StorageClass())
	writePickleString(pickle, key)
	writePickleString(pickle, "cpu")
	writePickleInt(pickle, count)
	pickle.WriteString("tQ") // TUPLE, BINPERSID
	writePickleInt(pickle, 0)
	for _, values := range [][]int64{shape, stride} {
		pickle.WriteByte('(')
		for _, value := range values {
			writePickleInt(pickle, value)
		}
		pickle.WriteByte('t')
	}
	pickle.Write([]byte{0x89, ')', 't', 'R'}) // requires_grad=False, no hooks, TUPLE, REDUCE
}

func writePickleString(pickle *bytes.Buffer, value string) {
	pickle.WriteByte('X')
	pickle.Write(binary.LittleEndian.AppendUint32(nil, uint32(len(value))))
	pickle.WriteString(value)
}

func writePickleGlobal(pickle *bytes.Buffer, module, name string) {
	pickle.WriteString("c" + module + "\n" + name + "\n")
}

func writePickleInt(pickle *bytes.Buffer, value int64) {
	pickle.WriteByte('J')
	pickle.Write(binary.LittleEndian.AppendUint32(nil, uint32(value)))
}
