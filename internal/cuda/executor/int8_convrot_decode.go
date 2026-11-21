package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// int8ConvRotLargestGroup is the widest column group the card's exact decode turns at once, 4^5.
const int8ConvRotLargestGroup = uint64(1024)

// Int8ConvRotDecode is an int8 ConvRot weight decoded on the card to the BF16 weight overgo's host decodes it to
// (comfyquant.DecodeRow), bit for bit: Stored is the weight as dtype.I8ConvRot holds it, Rows rows of Inner int8 values
// and then their F32 scales (comfyquant.Weight.Stored), Group the width of the column groups its rows are rotated in,
// and Output where its Rows rows of Inner BF16 values go.
type Int8ConvRotDecode struct {
	Stored             []byte
	Rows, Inner, Group uint32
	Output             driver.DevicePtr
}

// DecodeInt8ConvRot decodes the weight on the card, on the worker's turn (state): its int8 rows and scales are copied
// to the working memory the merges share (mergeWork) and decoded into the output (decode_int8_convrot_rows_exact_bf16),
// a block for each group of a row. The MOSS Ultra encoder's ConvRot weights had been decoded on the host, one core, 41 s
// of the workflow's first prompt, and copied to the card at twice the bytes (moss-first-run).
func (e *Executor) DecodeInt8ConvRot(state *device.State, decode Int8ConvRotDecode) error {
	rows, inner, group := uint64(decode.Rows), uint64(decode.Inner), uint64(decode.Group)
	elements := rows * inner
	if rows == 0 || !int8ConvRotDecodedGroup(group) || inner%group != 0 || elements/group > 1<<31-1 ||
		uint64(len(decode.Stored)) != elements+4*rows || decode.Output == 0 {
		return fmt.Errorf("int8 ConvRot decode of %d x %d in groups of %d over %d stored bytes is not a weight the decode takes", rows, inner, group, len(decode.Stored))
	}
	resources, err := e.ensureResources(state, false, 0, 0, 0)
	if err != nil {
		return err
	}
	work, err := e.mergeWork(state, uint64(len(decode.Stored)))
	if err != nil {
		return err
	}
	if err := state.Driver.MemcpyHtoD(work, decode.Stored); err != nil {
		return err
	}
	scales, output := work+driver.DevicePtr(elements), decode.Output
	innerArgument, rowsArgument, groupArgument := decode.Inner, decode.Rows, decode.Group
	return launchGridABI(state, resources.functions[kernelDecodeInt8ConvrotRowsExactBf16], kernel.Grid1D(int(elements/group)), kernel.DefaultBlock1D(),
		&work, &scales, &output, &innerArgument, &rowsArgument, &groupArgument)
}

// int8ConvRotDecodedGroup says the card's exact decode turns a group of that width: a power of 4 from 4 to 1024.
func int8ConvRotDecodedGroup(group uint64) bool {
	for width := uint64(4); width <= int8ConvRotLargestGroup; width *= 4 {
		if group == width {
			return true
		}
	}
	return false
}
