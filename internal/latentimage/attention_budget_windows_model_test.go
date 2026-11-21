//go:build modeltest && windows

package latentimage

import (
	"testing"
)

// attentionBlockLaunches are the launches a MOSS joint block's attention makes as comfy_kitchen's int8 attention, which
// the MOSS Ultra workflow's file declares for each joint layer: its key heads' anchors, Q and K quantized together, V
// quantized, and the attention over the codes, one each.
var attentionBlockLaunches = map[string]int{
	"int8_attention_anchor_f32":      1,
	"int8_attention_quantize_qk_f32": 1,
	"int8_attention_quantize_v_f32":  1,
	"int8_attention_bf16_f32":        1,
}

// bf16AttentionKernels are the BF16 attention kernels a pass may launch: the one with its scores in registers, which the
// text refinement's blocks run, as ComfyUI runs them on its default attention, the file declaring none for them, and
// the native and tiled kernels the fused BF16 attention falls back to.
var bf16AttentionKernels = []string{"attention_registers_bf16_f32", "attention_native_bf16_f32", "attention_tiled_bf16_f32"}

// TestMOSSAttentionMeetsItsBudget holds a MOSS pass's attention to ComfyUI's (moss-attention-pass): each joint block's
// attention is comfy_kitchen's int8 attention, ported bit for bit (TestTheInt8AttentionIsComfyKitchens), as the
// workflow's file declares it and ComfyUI runs it, and no joint block runs a BF16 attention. Counted over a text to
// image run (textToImageCensus): every block makes attentionBlockLaunches, and the BF16 attention launches only for the
// text refinement, its blocks once a pass, the step's two branches each a pass. Before, every block ran overgo's BF16
// attention with its scores in registers (attention_registers_bf16_f32), 26.6 ms of a pass at 1024 x 576 against
// ComfyUI's 11.53 ms for its int8 attention, and drew by other arithmetic than ComfyUI's.
func TestMOSSAttentionMeetsItsBudget(t *testing.T) {
	census, spec, blocks := textToImageCensus(t)
	for kernel, launches := range attentionBlockLaunches {
		if census[kernel] != launches*blocks {
			t.Errorf("%d blocks launched %s %d times; want %d a block", blocks, kernel, census[kernel], launches)
		}
	}
	if short := census["int8_attention_quantize_v_short_f32"]; short != 0 {
		t.Errorf("V's quantizer for 256 keys or fewer launched %d times over the workflow's thousands of keys", short)
	}
	bf16 := 0
	for _, kernel := range bf16AttentionKernels {
		bf16 += census[kernel]
	}
	t.Logf("%d joint blocks launched int8_attention_bf16_f32 %d times; the BF16 attention launched %d times", blocks, census["int8_attention_bf16_f32"], bf16)
	if passes := bf16 / spec.RefinerTextBlocks; bf16%spec.RefinerTextBlocks != 0 || passes < int8BudgetSteps || passes > 2*int8BudgetSteps {
		t.Errorf("the BF16 attention launched %d times over %d blocks; the text refinement's %d blocks launch it once a pass, %d to %d passes over %d steps",
			bf16, blocks, spec.RefinerTextBlocks, int8BudgetSteps, 2*int8BudgetSteps, int8BudgetSteps)
	}
}
