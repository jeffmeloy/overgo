package trainingworkflow

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

const (
	// changeTensorStorage is the storage a materialized change is written in:
	// the width a run's checkpoint holds its trained tensors at.
	changeTensorStorage = "F32"
	// changeDirectoryMode is the mode of the directory materialized changes
	// are written under.
	changeDirectoryMode = 0o755
)

// changeModelFiles are the files beside a model's weights that a materialized
// change carries over from its base, so its directory loads as the base does.
var changeModelFiles = []string{"config.json", "generation_config.json", "tokenizer.json"}

// changeTensors reads a base model and the runs trained from it one tensor at
// a time: each run's change is its tensor less the base's, so the changes of a
// whole model never sit in memory together.
type changeTensors struct {
	base  *safetensors.Source
	runs  []*safetensors.Source
	names []string
}

// openChangeTensors opens the base model's weights and each run's, and
// refuses a run that does not hold exactly the base's tensors.
func openChangeTensors(base string, runs []string) (opened *changeTensors, err error) {
	opened = &changeTensors{}
	defer func() {
		if err != nil {
			err = errors.Join(err, opened.Close())
		}
	}()
	if opened.base, err = safetensors.OpenSource(base); err != nil {
		return opened, err
	}
	opened.names = opened.base.Names()
	for _, directory := range runs {
		run, err := safetensors.OpenSource(directory)
		if err != nil {
			return opened, err
		}
		opened.runs = append(opened.runs, run)
		if len(run.Tensors) != len(opened.base.Tensors) {
			return opened, fmt.Errorf("training workflow: %s holds %d tensors, its base %d", directory, len(run.Tensors), len(opened.base.Tensors))
		}
		for _, name := range opened.names {
			if held, found := run.Tensors[name]; !found || held.Elements() != opened.base.Tensors[name].Elements() {
				return opened, fmt.Errorf("training workflow: %s does not hold the base's tensor %s", directory, name)
			}
		}
	}
	return opened, nil
}

// Close closes every opened source.
func (tensors *changeTensors) Close() error {
	var err error
	if tensors.base != nil {
		err = tensors.base.Close()
	}
	for _, run := range tensors.runs {
		err = errors.Join(err, run.Close())
	}
	return err
}

// change reads one tensor of the base and of every run and returns the base's
// values with each run's change from them. A tensor no run moved, as a frozen
// one is, returns no changes; it is found by reading the runs one at a time,
// so the largest frozen tensor costs two copies of itself and not one a run.
func (tensors *changeTensors) change(name string) (base []float32, changes [][]float32, err error) {
	if base, err = safetensors.ReadF32(tensors.base.Tensors[name]); err != nil {
		return nil, nil, err
	}
	changes = make([][]float32, len(tensors.runs))
	moved := false
	for index, run := range tensors.runs {
		trained, err := safetensors.ReadF32(run.Tensors[name])
		if err != nil {
			return nil, nil, err
		}
		for at, value := range trained {
			trained[at] = value - base[at]
			moved = moved || trained[at] != 0
		}
		if !moved {
			continue
		}
		// Runs read before the first that moved changed nothing: their changes are zero.
		for earlier := range index {
			if changes[earlier] == nil {
				changes[earlier] = make([]float32, len(base))
			}
		}
		changes[index] = trained
	}
	if !moved {
		return base, nil, nil
	}
	return base, changes, nil
}

// changeGram is the inner products of the runs' changes over every tensor,
// row-major: entry (i, j) is the change of run i dotted with the change of
// run j, so its diagonal holds each change's squared length. The products of
// one tensor are taken on every processor at once.
func changeGram(ctx context.Context, tensors *changeTensors) ([]float64, error) {
	count := len(tensors.runs)
	gram := make([]float64, count*count)
	for _, name := range tensors.names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		_, changes, err := tensors.change(name)
		if err != nil {
			return nil, err
		}
		if changes == nil {
			continue
		}
		var workers sync.WaitGroup
		rows := make(chan int)
		for range runtime.GOMAXPROCS(0) {
			workers.Go(func() {
				for row := range rows {
					for column := row; column < count; column++ {
						var product float64
						for at, value := range changes[row] {
							product += float64(value) * float64(changes[column][at])
						}
						// Each entry is written by the one worker that took its row.
						gram[row*count+column] += product
					}
				}
			})
		}
		for row := range count {
			rows <- row
		}
		close(rows)
		workers.Wait()
	}
	for row := range count {
		for column := range row {
			gram[row*count+column] = gram[column*count+row]
		}
	}
	return gram, nil
}

// materializeChanges writes, for each set of coefficients, the model that is
// the base plus that weighted sum of the runs' changes, as a directory that
// loads as the base does: its weights in one file and the base's
// configuration and tokenizer beside them. The directories are written
// together in one pass over the tensors.
func materializeChanges(ctx context.Context, tensors *changeTensors, baseDirectory string, directories []string, coefficients [][]float64) error {
	plan := safetensors.StreamingPlan{Shards: []safetensors.StreamingShardSpec{{Name: trainingprogram.CheckpointWeights}}}
	for _, name := range tensors.names {
		plan.Shards[0].Tensors = append(plan.Shards[0].Tensors, safetensors.StreamingTensorSpec{
			Name: name, DType: changeTensorStorage, Shape: tensors.base.Tensors[name].Shape,
		})
	}
	writers := make([]*safetensors.StreamingWriter, len(directories))
	for index, directory := range directories {
		if len(coefficients[index]) != len(tensors.runs) {
			return fmt.Errorf("training workflow: %d coefficients for %d runs", len(coefficients[index]), len(tensors.runs))
		}
		if err := os.MkdirAll(filepath.Dir(directory), changeDirectoryMode); err != nil {
			return err
		}
		var err error
		if writers[index], err = safetensors.NewStreamingWriter(directory, plan); err != nil {
			return err
		}
	}
	for _, name := range tensors.names {
		if err := ctx.Err(); err != nil {
			return err
		}
		base, changes, err := tensors.change(name)
		if err != nil {
			return err
		}
		for index, writer := range writers {
			payload := make([]byte, 0, len(base)*binaryschema.Uint32Bytes)
			for at, value := range base {
				sum := float64(value)
				for run, change := range changes {
					sum += coefficients[index][run] * float64(change[at])
				}
				if !checked.Finite32(float32(sum)) {
					return fmt.Errorf("training workflow: tensor %s of %s is not finite", name, directories[index])
				}
				payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(float32(sum)))
			}
			if err := writer.WriteTensor(name, uint64(len(payload)), bytes.NewReader(payload)); err != nil {
				return err
			}
		}
	}
	for index, writer := range writers {
		if err := writer.Finalize(); err != nil {
			return err
		}
		for _, file := range changeModelFiles {
			if err := copyModelFile(filepath.Join(baseDirectory, file), filepath.Join(directories[index], file)); err != nil {
				return err
			}
		}
	}
	return nil
}

// copyModelFile copies one of a base model's files beside a materialized
// change; a file the base does not have is not one the change needs.
func copyModelFile(from, to string) error {
	source, err := os.Open(from)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	target, err := os.Create(to)
	if err != nil {
		return errors.Join(err, source.Close())
	}
	_, err = io.Copy(target, source)
	return errors.Join(err, source.Close(), target.Close())
}
