package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

func main() {
	all := flag.Bool(
		"all",
		false,
		"convert eligible one-dimensional tensors as well as matrices",
	)
	imatrixPath := flag.String("imatrix", "", "pinned GGUF or legacy importance matrix")
	flag.Parse()
	if flag.NArg() != 3 {
		fmt.Fprintln(
			os.Stderr,
			"usage: gguf-quantize [-all] [-imatrix file] input.gguf output.gguf type",
		)
		os.Exit(2)
	}
	target, err := parseQuantizationType(flag.Arg(2))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	report, err := quantizeModel(flag.Arg(0), flag.Arg(1), target, *all, *imatrixPath)
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
	imatrixPath string,
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
	options := gguf.QuantizeOptions{}
	imatrixPath = strings.TrimSpace(imatrixPath)
	if quant.RequiresImportance(target) && imatrixPath == "" {
		return gguf.QuantizeReport{}, fmt.Errorf("gguf-quantize: %s requires -imatrix", target)
	}
	if imatrixPath != "" && !quant.RequiresImportance(target) {
		return gguf.QuantizeReport{}, fmt.Errorf("gguf-quantize: -imatrix is unsupported for %s", target)
	}
	if imatrixPath != "" {
		matrix, loadErr := gguf.LoadImportanceMatrix(imatrixPath)
		if loadErr != nil {
			return gguf.QuantizeReport{}, fmt.Errorf("gguf-quantize: load imatrix: %w", loadErr)
		}
		options.Importance = matrix.Entries
		options.ImportanceFile = imatrixPath
		options.ImportanceDatasets = matrix.Datasets
		options.ImportanceChunkCount = matrix.ChunkCount
	}
	if all {
		options.ShouldQuantize = eligibleTensor
	}

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
	if dataType, ok := quant.ParseType(value); ok {
		return dataType, nil
	}
	return 0, fmt.Errorf(
		"gguf-quantize: unsupported type %q (use %s)",
		value, strings.Join(quant.TypeNames(), ", "),
	)
}
