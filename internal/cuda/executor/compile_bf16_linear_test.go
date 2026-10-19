package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestCompileBF16LinearBoundaries(t *testing.T) {
	for _, mode := range []string{"rounded", "biasless", "biasless-retained", "biasless-shared", "unrounded-bias", "retained-product", "retained-add", "shared-product"} {
		t.Run(mode, func(t *testing.T) {
			b := tensor.NewBuilder()
			b.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
			w := b.Input("weight", dtype.BF16, tensor.MustShape(8, 16))
			x := b.Input("input", dtype.F32, tensor.MustShape(8, 4))
			bias := b.Input("bias", dtype.F32, tensor.MustShape(16))
			if mode != "unrounded-bias" {
				bias = b.BF16Round(bias)
			}
			product := b.MulMat(w, x)
			add := b.Add(product, bias)
			output := b.BF16Round(add)
			if mode == "biasless" || mode == "biasless-retained" || mode == "biasless-shared" {
				output = b.BF16Round(product)
			}
			outputs := []*tensor.Tensor{output}
			switch mode {
			case "retained-product", "biasless-retained":
				outputs = append(outputs, product)
			case "retained-add":
				outputs = append(outputs, add)
			case "shared-product", "biasless-shared":
				outputs = append(outputs, b.Scale(product, 2))
			}
			compiled, err := Compile(outputs...)
			if err != nil {
				t.Fatal(err)
			}
			fusion := compiled.nodes[compiled.orderIndexes[output]].fusion
			if got := fusion != nil && fusion.kind == compiledFusionBF16Linear; got != (mode == "rounded" || mode == "biasless") {
				t.Fatalf("BF16 linear fusion=%v", got)
			}
			if mode == "rounded" || mode == "biasless" {
				want, err := bf16LinearStagingBytes([]*tensor.Tensor{w, x, bias})
				if err != nil || compiled.matmulStagingBytes != want || !compiled.needBlas {
					t.Fatalf("shared staging=%d want=%d error=%v", compiled.matmulStagingBytes, want, err)
				}
			}
		})
	}
}
