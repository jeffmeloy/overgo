package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/gguf"
)

// mergeCommand is gguf merge: it merges split GGUF shards into one file.
func mergeCommand(flags *flag.FlagSet, args []string) error {
	output := flags.String("out", "", "new single-file GGUF output path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 || *output == "" {
		return fmt.Errorf("usage: %s -out output.gguf input.gguf", flags.Name())
	}
	return merge(flags.Arg(0), *output)
}

func merge(inputPath, outputPath string) error {
	inputAbsolute, err := filepath.Abs(inputPath)
	if err != nil {
		return err
	}
	outputAbsolute, err := filepath.Abs(outputPath)
	if err != nil {
		return err
	}
	if strings.EqualFold(filepath.Clean(inputAbsolute), filepath.Clean(outputAbsolute)) {
		return errors.New("gguf merge:input and output paths are identical")
	}
	source, err := gguf.Open(inputAbsolute)
	if err != nil {
		return fmt.Errorf("gguf merge:open input: %w", err)
	}
	defer source.Close()

	destination, err := os.OpenFile(
		outputAbsolute,
		os.O_WRONLY|os.O_CREATE|os.O_EXCL,
		0o644,
	)
	if err != nil {
		return fmt.Errorf("gguf merge:create output: %w", err)
	}
	succeeded := false
	defer func() {
		_ = destination.Close()
		if !succeeded {
			_ = os.Remove(outputAbsolute)
		}
	}()
	if err := source.WriteTo(destination, gguf.WriteOptions{}); err != nil {
		return fmt.Errorf("gguf merge:write output: %w", err)
	}
	if err := destination.Sync(); err != nil {
		return fmt.Errorf("gguf merge:sync output: %w", err)
	}
	if err := destination.Close(); err != nil {
		return fmt.Errorf("gguf merge:close output: %w", err)
	}
	succeeded = true
	return nil
}
