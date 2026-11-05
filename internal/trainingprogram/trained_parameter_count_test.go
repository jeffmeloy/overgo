package trainingprogram

import (
	"math"
	"testing"

	"overgo/internal/testcheck"
)

// TestRateReadsThePlansParameterCount pins the rate Qwen2.5-0.5B runs at, from every parameter of its plan, its frozen
// lexical table among them, against the higher rate of counting only what it trains, which held out worse (OptimizerPolicy).
func TestRateReadsThePlansParameterCount(t *testing.T) {
	t.Parallel()
	const plan, tolerance, measured = 494032768, 1e-3, 6.078e-06
	config, err := BuiltinOptimizerPolicy().Config(plan)
	testcheck.NoError(t, err)
	if math.Abs(config.BaseLearningRate-measured) > measured*tolerance {
		t.Fatalf("the plan's rate = %g, want the %g that was judged", config.BaseLearningRate, measured)
	}
}
