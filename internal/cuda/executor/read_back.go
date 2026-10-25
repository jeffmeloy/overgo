package executor

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// ReadBack names outputs that every retained run copies to host memory behind its graph, before the one
// synchronization the run ends with, so RetainedOutputs.CopyToHost returns them with no further wait on the card.
//   - a speculative window's selections and hidden rows, and its draft's token, each took a round trip to the card's
//     worker and a blocking copy after the run, about 0.25 ms of a Flash-Next window.
func (c *CompiledGraph) ReadBack(outputs ...*tensor.Tensor) error {
	if c == nil {
		return errors.New("CUDA compiled graph is nil")
	}
	readBack := make([]bool, len(c.outputs))
	for _, output := range outputs {
		index, ok := c.outputIndexes[output]
		if !ok || output.Type != dtype.F32 {
			return fmt.Errorf("read-back output %q is not an F32 output of the compiled graph", output.Name)
		}
		readBack[index] = true
	}
	c.readBack = readBack
	return nil
}

// readBackCopy is one read-back output's place in the pinned read-back buffer.
type readBackCopy struct {
	index      int
	at, length uint64
}

// planReadBack places a retained run's read-back outputs in the pinned read-back buffer, growing it, before the run
// is launched: a pinned allocation can wait for the card, whose run may wait for this worker (moeDoorbell).
func (c *CompiledGraph) planReadBack(state *device.State, values []DeviceValue, staging *[]byte) ([]readBackCopy, error) {
	var copies []readBackCopy
	var total uint64
	for index, read := range c.readBack {
		if !read || values[index].Pointer == 0 {
			continue
		}
		length, err := values[index].Shape.Bytes(dtype.F32)
		if err != nil {
			return nil, err
		}
		copies = append(copies, readBackCopy{index: index, at: total, length: length})
		total += moeAlign(length)
	}
	if len(copies) == 0 {
		return nil, nil
	}
	return copies, pinned(state, staging, total)
}

// queueReadBack queues the planned copies behind the launched run.
func queueReadBack(state *device.State, copies []readBackCopy, values []DeviceValue, staging []byte) error {
	for _, read := range copies {
		if err := state.Driver.MemcpyDtoHAsync(staging[read.at:read.at+read.length], values[read.index].Pointer, state.Stream); err != nil {
			return err
		}
	}
	return nil
}

// takeReadBack copies the read-back outputs out of pinned memory once the run's stream is synchronized, one slice
// an output and nil for the others.
func takeReadBack(copies []readBackCopy, staging []byte, outputs int) [][]float32 {
	if len(copies) == 0 {
		return nil
	}
	read := make([][]float32, outputs)
	for _, copied := range copies {
		values := make([]float32, copied.length/f32ScalarBytes)
		copy(driver.Bytes(values), staging[copied.at:copied.at+copied.length])
		read[copied.index] = values
	}
	return read
}
