package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

func main() {
	all := flag.Bool(
		"all",
		false,
		"convert eligible one-dimensional tensors as well as matrices",
	)
	flag.Parse()
	if flag.NArg() != 3 {
		fmt.Fprintln(
			os.Stderr,
			"usage: gguf-quantize [-all] input.gguf output.gguf type",
		)
		os.Exit(2)
	}
	target, err := parseQuantizationType(flag.Arg(2))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	report, err := quantizeModel(flag.Arg(0), flag.Arg(1), target, *all)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf(
		"converted %d tensors, preserved %d; tensor bytes %d -> %d\n",
		report.Converted,
		report.Preserved,
		report.InputBytes,
		report.OutputBytes,
	)
}

func quantizeModel(
	inputPath, outputPath string,
	target dtype.Type,
	all bool,
) (gguf.QuantizeReport, error) {
	inputAbsolute, err := filepath.Abs(inputPath)
	if err != nil {
		return gguf.QuantizeReport{}, err
	}
	outputAbsolute, err := filepath.Abs(outputPath)
	if err != nil {
		return gguf.QuantizeReport{}, err
	}
	if strings.EqualFold(filepath.Clean(inputAbsolute), filepath.Clean(outputAbsolute)) {
		return gguf.QuantizeReport{}, errors.New(
			"gguf-quantize: input and output paths are identical",
		)
	}
	source, err := gguf.Open(inputAbsolute)
	if err != nil {
		return gguf.QuantizeReport{}, fmt.Errorf(
			"gguf-quantize: open input: %w",
			err,
		)
	}
	defer source.Close()

	destination, err := os.OpenFile(
		outputAbsolute,
		os.O_WRONLY|os.O_CREATE|os.O_EXCL,
		0o644,
	)
	if err != nil {
		return gguf.QuantizeReport{}, fmt.Errorf(
			"gguf-quantize: create output: %w",
			err,
		)
	}
	succeeded := false
	defer func() {
		_ = destination.Close()
		if !succeeded {
			_ = os.Remove(outputAbsolute)
		}
	}()
	options := gguf.QuantizeOptions{}
	if all {
		options.ShouldQuantize = eligibleTensor
	}
	report, err := source.QuantizeTo(destination, target, options)
	if err != nil {
		return report, fmt.Errorf("gguf-quantize: write output: %w", err)
	}
	if err := destination.Sync(); err != nil {
		return report, fmt.Errorf("gguf-quantize: sync output: %w", err)
	}
	if err := destination.Close(); err != nil {
		return report, fmt.Errorf("gguf-quantize: close output: %w", err)
	}
	succeeded = true
	return report, nil
}

func eligibleTensor(tensor gguf.TensorInfo) bool {
	traits, ok := tensor.Type.Traits()
	if !ok {
		return false
	}
	switch tensor.Type {
	case dtype.F32, dtype.F16, dtype.BF16:
		return true
	default:
		return traits.Quantized
	}
}

func parseQuantizationType(value string) (dtype.Type, error) {
	normalized := strings.ToLower(strings.TrimSpace(value))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	types := map[string]dtype.Type{
		"f32":     dtype.F32,
		"f16":     dtype.F16,
		"bf16":    dtype.BF16,
		"q1_0":    dtype.Q1_0,
		"q2_0":    dtype.Q2_0,
		"q4_0":    dtype.Q4_0,
		"q4_1":    dtype.Q4_1,
		"q5_0":    dtype.Q5_0,
		"q5_1":    dtype.Q5_1,
		"q8_0":    dtype.Q8_0,
		"q2_k":    dtype.Q2K,
		"q3_k":    dtype.Q3K,
		"q4_k":    dtype.Q4K,
		"q5_k":    dtype.Q5K,
		"q6_k":    dtype.Q6K,
		"tq1_0":   dtype.TQ1_0,
		"tq2_0":   dtype.TQ2_0,
		"mxfp4":   dtype.MXFP4,
		"nvfp4":   dtype.NVFP4,
		"iq4_nl":  dtype.IQ4NL,
		"iq4_xs":  dtype.IQ4XS,
		"iq2_s":   dtype.IQ2S,
		"iq3_xxs": dtype.IQ3XXS,
		"iq3_s":   dtype.IQ3S,
	}
	if dataType, ok := types[normalized]; ok {
		return dataType, nil
	}
	return 0, fmt.Errorf(
		"gguf-quantize: unsupported type %q (use f32, f16, bf16, q1_0, q2_0, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k, tq1_0, tq2_0, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, mxfp4, or nvfp4)",
		value,
	)
}
