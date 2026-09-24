// Package gguf_test validates GGUF against an independently serialized fixture.
package gguf_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gguf"
)

func TestPendingFixtureTinyOracles(t *testing.T) {
	root := filepath.Join("..", "parity", "testdata")
	raw, err := os.ReadFile(filepath.Join(root, "olmoe_tiny.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	const pinnedSHA = "191e65fe8477cd1e50a9d14ee832aa6bf10b39980b61189831f2f44f2a05a5d1"
	actual := sha256.Sum256(raw)
	if hex.EncodeToString(actual[:]) != pinnedSHA {
		t.Fatalf("tiny GGUF source changed: %x", actual)
	}
	manifestBytes, err := os.ReadFile(filepath.Join(root, "olmoe_tiny.json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Format       string `json:"format"`
		Architecture string `json:"architecture"`
		SHA256       string `json:"sha256"`
		Tensors      []struct {
			Name   string   `json:"name"`
			Shape  []uint64 `json:"shape"`
			SHA256 string   `json:"sha256"`
		} `json:"tensors"`
		QRowSums []float64 `json:"q_row_sums"`
	}
	if err := json.Unmarshal(manifestBytes, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Format != "GGUF-v3-independent-python" || oracle.Architecture != "olmoe" || oracle.SHA256 != pinnedSHA {
		t.Fatalf("unexpected independent fixture provenance: %+v", oracle)
	}
	file, err := gguf.Parse(bytes.NewReader(raw), uint64(len(raw)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	var rewritten bytes.Buffer
	if err := file.WriteTo(&rewritten, gguf.WriteOptions{}); err != nil || !bytes.Equal(rewritten.Bytes(), raw) {
		t.Fatalf("Go GGUF writer differs from the independent Python wire fixture: %v", err)
	}
	architecture, ok := file.MetadataValue("general.architecture")
	if !ok || architecture.Data != oracle.Architecture || len(file.Tensors) != len(oracle.Tensors) || len(file.Tensors) != 15 {
		t.Fatalf("tiny OLMoE metadata/inventory differs: architecture=%+v tensors=%d", architecture, len(file.Tensors))
	}
	for index, expected := range oracle.Tensors {
		info := file.Tensors[index]
		if info.Name != expected.Name || info.Type != gguf.DTypeF32 || int(info.Dimensions) != len(expected.Shape) ||
			!slices.Equal(info.Shape[:info.Dimensions], expected.Shape) {
			t.Fatalf("tensor %d = %+v, want %+v", index, info, expected)
		}
		payload := make([]byte, info.Size)
		if err := file.ReadTensorData(info, payload); err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(payload)
		if hex.EncodeToString(digest[:]) != expected.SHA256 {
			t.Fatalf("tensor %q payload differs", info.Name)
		}
		if info.Name == "blk.0.attn_q.weight" {
			if len(oracle.QRowSums) != 8 || len(payload) != 8*8*4 {
				t.Fatal("Q projection oracle shape differs")
			}
			for row := range oracle.QRowSums {
				var sum float64
				for column := range 8 {
					offset := (row*8 + column) * 4
					sum += float64(math.Float32frombits(binary.LittleEndian.Uint32(payload[offset:])))
				}
				if sum != oracle.QRowSums[row] {
					t.Fatalf("Q row %d sum=%g, want independent %g", row, sum, oracle.QRowSums[row])
				}
			}
		}
	}
}
