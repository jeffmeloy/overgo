//go:build windows

package executor

import (
	"context"
	"testing"

	cudatest "llamacpp2go/internal/cuda/testutil"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

type graphOutputCheck struct {
	output    *tensor.Tensor
	tolerance float64
}

func checkCUDAGraph(
	t *testing.T,
	feeds map[*tensor.Tensor]reference.Value,
	checks ...graphOutputCheck,
) {
	t.Helper()
	cudatest.Require(t)
	outputs := make([]*tensor.Tensor, len(checks))
	for index, check := range checks {
		outputs[index] = check.output
	}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, check := range checks {
		compare(t, got[check.output].Data, want[check.output].Data, check.tolerance)
	}
}

func uniformGraphChecks(outputs []*tensor.Tensor, tolerance float64) []graphOutputCheck {
	checks := make([]graphOutputCheck, len(outputs))
	for index, output := range outputs {
		checks[index] = graphOutputCheck{output: output, tolerance: tolerance}
	}
	return checks
}
