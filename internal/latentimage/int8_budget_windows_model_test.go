//go:build modeltest && windows

package latentimage

import (
	"context"
	"maps"
	"slices"
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// int8BudgetSteps are the steps textToImageCensus draws, the fewest MOSS's schedule draws.
const int8BudgetSteps = 2

// int8BlockLaunches are the launches of each int8 kernel a MOSS joint block makes, the block counted by its attention
// output's quantization (quantize_int8_convrot_columns_f32), which it makes once: the q and k products normalizing and
// rotating their heads in their epilogue, the v and feed-up products writing their scaled entries, the output and
// feed-down projections writing their gated sums, the block's modulated norm quantized once for q, k and v, and the
// feed's norm and the feed-down's SiTU each quantized as they are computed.
var int8BlockLaunches = map[string]int{
	"int8_convrot_product_norm_rotate_bf16_f32": 2,
	"int8_convrot_product_bf16_f32":             2,
	"int8_convrot_product_gated_add_bf16_f32":   2,
	"quantize_int8_convrot_columns_f32":         1,
	"quantize_int8_convrot_norm_columns_f32":    2,
	"quantize_int8_convrot_situ_columns_f32":    1,
}

// textToImageCensus counts each kernel's launches, by the executor's launch census, over the MOSS Ultra workflow's text
// to image section run for int8BudgetSteps steps at the workflow's size, and gives the transformer it ran and the joint
// blocks it ran, each counted by its attention output's quantization (quantize_int8_convrot_columns_f32), which a block
// makes once.
func textToImageCensus(t *testing.T) (map[string]int, TransformerSpec, int) {
	t.Helper()
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	var request Request
	for _, section := range workflowSections(t) {
		if section.slug == "text-to-image" {
			request = section.request(t, readComfySection(t, "pass-2", section.slug))
		}
	}
	request.Steps = int8BudgetSteps
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	session := testcheck.Must(generator.prepare(ctx, request))(t)
	census := map[string]int{}
	runtime := generator.rotary.runtime
	testcheck.NoError(t, runtime.Do(ctx, func(state *device.State) error {
		state.Trace = &device.LaunchTrace{Census: census}
		return nil
	}))
	_, err := generator.integrate(ctx, session)
	testcheck.NoError(t, runtime.Do(context.WithoutCancel(ctx), func(state *device.State) error {
		state.Trace = nil
		return nil
	}))
	testcheck.NoError(t, err)
	// A guided step runs each block once for each branch, the negative's skipped layers apart.
	spec := generator.rotary.transformer
	blocks := census["quantize_int8_convrot_columns_f32"]
	if blocks < int8BudgetSteps*spec.Layers || blocks > 2*int8BudgetSteps*spec.Layers {
		t.Fatalf("%d steps quantized %d blocks' attention outputs; a step runs %d layers for each of its two branches", int8BudgetSteps, blocks, spec.Layers)
	}
	return census, spec, blocks
}

// TestMOSSInt8ProductsMeetTheirBudget holds a MOSS pass's int8 products to comfy_kitchen's form of them
// (moss-int8-epilogue): each product is one launch of a kernel that applies its scales in its epilogue, as
// comfy_kitchen's CUTLASS product applies them, and writes what the pass reads of it (int8_convrot_product_bf16_f32 and
// its gated sum and head norm), with each activation quantized once beside it. Counted over a text to image run
// (textToImageCensus): every block makes int8BlockLaunches and no int8 kernel launches beside them but the attention's
// (attentionBlockLaunches). Before, cuBLASLt's int8 product wrote an int32 product that a separate kernel read back to
// scale (scale_int8_product_bf16_f32 and its gated sum and head norm), 16.8 ms of a pass's 114.6 ms of kernels, and the
// products and their epilogues took 66.5 ms against ComfyUI's 45.1.
func TestMOSSInt8ProductsMeetTheirBudget(t *testing.T) {
	census, _, blocks := textToImageCensus(t)
	for kernel, launches := range int8BlockLaunches {
		if census[kernel] != launches*blocks {
			t.Errorf("%d blocks launched %s %d times; want %d a block", blocks, kernel, census[kernel], launches)
		}
	}
	for _, kernel := range slices.Sorted(maps.Keys(census)) {
		_, counted := int8BlockLaunches[kernel]
		if _, attention := attentionBlockLaunches[kernel]; strings.Contains(kernel, "int8") && !counted && !attention {
			t.Errorf("%s launched %d times beside the products that apply their scales in their epilogue", kernel, census[kernel])
		}
	}
}
