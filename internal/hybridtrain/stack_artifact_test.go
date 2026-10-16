package hybridtrain

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
)

func TestLoadStackFeedForward(t *testing.T) {
	for _, storageType := range []dtype.Type{dtype.F32, dtype.BF16} {
		t.Run(storageType.String(), func(t *testing.T) {
			names := []string{"gate", "up", "down", "norm"}
			var tensors []gguf.TensorData
			var expected [][]float32
			for index, name := range names {
				values := []float32{float32(index+1) * 10, float32(index+1)*10 + .25, float32(index+1)*10 + .5, float32(index+1)*10 + .75}
				var raw []byte
				for _, value := range values {
					if storageType == dtype.F32 {
						raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(value))
					} else {
						raw = binary.LittleEndian.AppendUint16(raw, dtype.Float32ToBF16(value))
					}
				}
				tensors = append(tensors, gguf.TensorData{Name: name, Shape: []uint64{2, 2}, Type: storageType, Data: bytes.NewReader(raw)})
				expected = append(expected, values)
			}
			file, err := gguf.Open(testutil.TempGGUF(t, "layer.gguf", nil, tensors))
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			info := model.LayerWeights{FeedForwardGate: &file.Tensors[0], FeedForwardUp: &file.Tensors[1], FeedForwardDown: &file.Tensors[2], AttentionNorm: &file.Tensors[3]}
			m := &Model{matW: make([]float32, 2, 14)}
			m.matW[0], m.matW[1] = 101, 102
			masters := m.matW[len(m.matW):cap(m.matW)]
			var scratch []byte
			host, err := loadStackLayer(t.Context(), file, info, masters, len(expected[0]), &scratch)
			if err != nil {
				t.Fatal(err)
			}
			b := &builder{m: m}
			var bound [3][]float32
			for index, value := range []*reference.Value{host.FeedForwardGate, host.FeedForwardUp, host.FeedForwardDown} {
				if value == nil || !slices.Equal(value.Data, expected[index]) || &value.Data[0] != &masters[index*4] {
					t.Fatalf("matrix %d is not decoded directly into its master slot: %+v", index, value)
				}
				if err := b.matValues(&bound[index], names[index], 2, 2, value.Data); err != nil {
					t.Fatal(err)
				}
			}
			b.rebindAliases()
			want := slices.Concat([]float32{101, 102}, expected[0], expected[1], expected[2])
			if !slices.Equal(m.matW, want) || !slices.Equal(host.AttentionNorm.Data, expected[3]) {
				t.Fatalf("master order or remaining layer tensor changed: %v / %v", m.matW, host.AttentionNorm.Data)
			}
			for index := range bound {
				if &bound[index][0] != &masters[index*4] {
					t.Fatalf("binding %d copied out of the master slab", index)
				}
			}
			bad := *info.FeedForwardGate
			bad.Shape[0]++
			info.FeedForwardGate = &bad
			if _, err := loadStackLayer(t.Context(), file, info, masters, len(expected[0]), &scratch); err == nil {
				t.Fatal("wrong matrix extent accepted")
			}
			info.FeedForwardGate = &file.Tensors[0]
			info.AttentionNorm = nil
			cancelled, cancel := context.WithCancel(t.Context())
			cancel()
			if _, err := loadStackLayer(cancelled, file, info, masters, len(expected[0]), &scratch); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled decode: %v", err)
			}
		})
	}
}
