//go:build modeltest && windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// refinedSteps are the steps each section of TestThePromptIsRefinedOnceARequest draws each way.
const refinedSteps = 2

// TestThePromptIsRefinedOnceARequest holds moss-pass-repeats/refine (SQA-422): a MOSS sampling refines each branch's
// text once, by its text program (rotaryTextProgram), and its passes read the rows: the pass program reads none of the
// refiner's weights, the text program reads every one the pass had read, and text to image and depth control (its
// adapters patching the refiner's weights, moss-control and moss_image_sft among them), each drawn refinedSteps steps
// as the workflow's pass-2 recording asks it, end in the final latent the passes that refine the text draw, bit for bit.
// The refiner reads the prompt and the text's rotary rows alone, no timestep and no latent, and had run in each of a
// 50-step guided request's 100 passes.
func TestThePromptIsRefinedOnceARequest(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	ultraDepth(t, root)
	profile := testcheck.Must(ResolveProfile(root))(t)
	refines := samplingRefinesOnce
	t.Cleanup(func() { samplingRefinesOnce = refines })
	var generator *Generator
	defer func() {
		if generator != nil {
			closeReporting(t, context.WithoutCancel(ctx), generator)
		}
	}()
	checked := false
	for _, section := range workflowSections(t) {
		if section.slug != "text-to-image" && section.slug != "depth-control" {
			continue
		}
		recording := readComfySection(t, "pass-2", section.slug)
		request := section.request(t, recording)
		if section.slug == "depth-control" {
			request, _ = depthControlSection(t, recording)
		}
		request.Steps = refinedSteps
		finals := map[bool]string{}
		for _, once := range []bool{false, true} {
			samplingRefinesOnce = once
			if generator == nil || generator.Reset(ctx, request) != nil {
				if generator != nil {
					closeReporting(t, context.WithoutCancel(ctx), generator)
				}
				generator = testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
			}
			session := testcheck.Must(generator.prepare(ctx, request))(t)
			testcheck.Must(generator.integrate(ctx, session))(t)
			final := testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t)
			testcheck.Must(generator.decode(ctx, session))(t)
			finals[once] = latentBits(final.Data)
		}
		if finals[true] != finals[false] {
			t.Errorf("%s's final latent after %d steps hashes %s with its text refined once; the passes that refine it drew %s",
				section.slug, refinedSteps, finals[true], finals[false])
		}
		if !checked {
			checked = true
			refinerWeightsMoved(t, generator.rotary.transformer)
		}
	}
	if !checked {
		t.Fatal("the workflow names no text to image or depth control section")
	}
}

// refinerWeightsMoved holds the programs' weights: the pass that takes its text refined reads none of the refiner's
// (text_embedder), and the text program reads every weight of the refiner the pass that refines had read.
func refinerWeightsMoved(t *testing.T, spec TransformerSpec) {
	t.Helper()
	const height, width, text = 2, 3, 4
	refining := testcheck.Must(compileRotaryDenoiserProgram(spec, mossRefinementEpsilon, height, width, text, nil, dtype.BF16, rotaryOptions{}))(t)
	passing := testcheck.Must(compileRotaryDenoiserProgram(spec, mossRefinementEpsilon, height, width, text, nil, dtype.BF16, rotaryOptions{refinedText: true}))(t)
	refiner := testcheck.Must(compileRotaryTextProgram(spec, mossRefinementEpsilon, height, width, text, dtype.BF16, rotaryOptions{}))(t)
	held := func(weights tensor.WeightBindings) map[string]bool {
		named := map[string]bool{}
		for _, node := range weights {
			if strings.HasPrefix(node.Name, "text_embedder.") {
				named[node.Name] = true
			}
		}
		return named
	}
	before, pass, after := held(refining.weightInputs), held(passing.weightInputs), held(refiner.weightInputs)
	if len(before) == 0 || len(pass) != 0 {
		t.Fatalf("the refiner's weights: the refining pass reads %d, the pass that takes its text refined %d; want some and none", len(before), len(pass))
	}
	for name := range before {
		if !after[name] {
			t.Errorf("the text program reads no %s, which the refining pass read", name)
		}
	}
}

// latentBits is the SHA-256 of values' F32 bits, little-endian.
func latentBits(values []float32) string {
	bits := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(bits[4*index:], math.Float32bits(value))
	}
	sum := sha256.Sum256(bits)
	return hex.EncodeToString(sum[:])
}
