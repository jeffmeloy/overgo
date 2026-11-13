package driver

import (
	"errors"
	"testing"

	"overgo/internal/envsetting"
)

// cudaTestEnvironment is the switch the CUDA lane sets for a test run that may reach the card (cudatest.Require).
const cudaTestEnvironment = "OVERGO_CUDA_TEST"

// hostTestReach refuses a context to a test run without the CUDA lane's switch: a host test that reaches the card waits
// behind whatever holds it, unmeasured, in a landing's host stages, where on 2026-11-12 six test binaries waited 26
// minutes behind two renders (card-reach). Such a test is gated with cudatest.Require, and the CUDA lane runs it under
// the landing's one turn. A program that is no test is not refused.
func hostTestReach() error {
	if !testing.Testing() || envsetting.Neutral(cudaTestEnvironment, "admits a test to the card; it changes no result") == "1" {
		return nil
	}
	return errors.New("cuda: a test reached the card without " + cudaTestEnvironment + "=1; gate it with cudatest.Require, so the CUDA lane runs it under one turn")
}
