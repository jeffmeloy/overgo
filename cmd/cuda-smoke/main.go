package main

import (
	"context"
	"fmt"
	"os"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/kernel"
)

func run() error {
	worker, err := device.New(0)
	if err != nil {
		return err
	}
	defer worker.Close()

	inputA := []float32{1, 2, 3, 4}
	inputB := []float32{10, 20, 30, 40}
	var output []float32
	if err := worker.Do(context.Background(), func(state *device.State) error {
		var launchErr error
		output, launchErr = kernel.VectorAdd(state.Driver, state.Stream, inputA, inputB)
		return launchErr
	}); err != nil {
		return err
	}
	fmt.Println(output)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "cuda-smoke:", err)
		os.Exit(1)
	}
}
