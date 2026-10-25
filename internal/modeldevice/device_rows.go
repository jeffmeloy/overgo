package modeldevice

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

// placeRowsChunk is the source rows a placement reads from the file at once.
const placeRowsChunk = 4096

// PlaceRows places a tensor of the source matrix's rows, in the order rows names them, under name, read from the
// file once and stored as the file stores them; it returns the placed tensor's descriptor.
//   - a draft's head over a vocabulary subset takes its rows of the trunk's head this way, as Strata's MTP draft head
//     gathers rt/draft_vocab.bin's rows (core/mtp.cpp), so a draft step reads its share of the head's bytes.
func (w *DeviceWeights) PlaceRows(ctx context.Context, file *gguf.File, source gguf.TensorInfo, name string, rows []uint32) (gguf.TensorInfo, error) {
	if w == nil || w.deviceTensorStore == nil || file == nil {
		return gguf.TensorInfo{}, errors.New("device weights: rows have no store or file")
	}
	if source.Dimensions != tensor.PairedExtent || len(rows) == 0 || source.Shape[1] == 0 || source.Size%source.Shape[1] != 0 {
		return gguf.TensorInfo{}, fmt.Errorf("device weights: tensor %q has no rows to place", source.Name)
	}
	if _, ok := w.Lookup(name); ok {
		return gguf.TensorInfo{}, fmt.Errorf("device weights: tensor %q is already placed", name)
	}
	count, rowBytes := source.Shape[1], source.Size/source.Shape[1]
	// The rows are read in file order, a chunk at a time, each copied to every place the order names it.
	order := make([]int, len(rows))
	for index, row := range rows {
		if uint64(row) >= count {
			return gguf.TensorInfo{}, fmt.Errorf("device weights: row %d of tensor %q exceeds its %d rows", row, source.Name, count)
		}
		order[index] = index
	}
	slices.SortFunc(order, func(a, b int) int { return int(rows[a]) - int(rows[b]) })
	placed := make([]byte, uint64(len(rows))*rowBytes)
	chunk := make([]byte, placeRowsChunk*rowBytes)
	next := 0
	for first := uint64(0); first < count && next < len(order); first += placeRowsChunk {
		end := min(first+placeRowsChunk, count)
		if uint64(rows[order[next]]) >= end {
			continue
		}
		read := chunk[:(end-first)*rowBytes]
		if err := file.ReadTensorRange(source, first*rowBytes, read); err != nil {
			return gguf.TensorInfo{}, err
		}
		for ; next < len(order) && uint64(rows[order[next]]) < end; next++ {
			at := (uint64(rows[order[next]]) - first) * rowBytes
			copy(placed[uint64(order[next])*rowBytes:], read[at:at+rowBytes])
		}
	}
	info := gguf.TensorInfo{Name: name, Dimensions: tensor.PairedExtent, Type: source.Type, Size: uint64(len(placed))}
	info.Shape[0], info.Shape[1] = source.Shape[0], uint64(len(rows))
	return info, w.place(ctx, info, placed)
}
