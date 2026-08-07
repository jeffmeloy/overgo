package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"overgo/internal/gguf"
)

func main() {
	outputPrefix := flag.String("out-prefix", "", "output path prefix")
	maxTensors := flag.Int("max-tensors", 128, "maximum tensors per shard")
	maxSize := flag.String("max-size", "", "target tensor bytes per shard (K, M, or G suffix)")
	noTensorFirst := flag.Bool(
		"no-tensor-first-split",
		false,
		"write metadata-only first shard",
	)
	flag.Parse()
	if flag.NArg() != 1 || *outputPrefix == "" {
		fmt.Fprintln(
			os.Stderr,
			"usage: gguf-split -out-prefix prefix [-max-tensors 128] [-max-size 4G] input.gguf",
		)
		os.Exit(2)
	}
	bytes, err := parseSize(*maxSize)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := split(
		flag.Arg(0),
		*outputPrefix,
		gguf.SplitOptions{
			MaxTensors:           *maxTensors,
			MaxBytes:             bytes,
			NoTensorsInFirstFile: *noTensorFirst,
		},
	); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func split(inputPath, outputPrefix string, options gguf.SplitOptions) error {
	source, err := gguf.Open(inputPath)
	if err != nil {
		return fmt.Errorf("gguf-split: open input: %w", err)
	}
	defer source.Close()

	created := make([]string, 0)
	succeeded := false
	defer func() {
		if succeeded {
			return
		}
		for _, path := range created {
			_ = os.Remove(path)
		}
	}()
	err = source.WriteSplit(
		func(index, count uint16) (io.WriteCloser, error) {
			path := fmt.Sprintf(
				"%s-%05d-of-%05d.gguf",
				outputPrefix,
				int(index)+1,
				count,
			)
			file, createErr := os.OpenFile(
				path,
				os.O_WRONLY|os.O_CREATE|os.O_EXCL,
				0o644,
			)
			if createErr != nil {
				return nil, createErr
			}
			created = append(created, path)
			return &syncedFile{File: file}, nil
		},
		options,
	)
	if err != nil {
		return fmt.Errorf("gguf-split: %w", err)
	}
	succeeded = true
	return nil
}

type syncedFile struct {
	*os.File
}

func (f *syncedFile) Close() error {
	return errors.Join(f.Sync(), f.File.Close())
}

func parseSize(value string) (uint64, error) {
	value = strings.TrimSpace(value)
	original := value
	if value == "" {
		return 0, nil
	}
	multiplier := uint64(1)
	switch suffix := value[len(value)-1]; suffix {
	case 'k', 'K':
		multiplier = 1 << 10
		value = value[:len(value)-1]
	case 'm', 'M':
		multiplier = 1 << 20
		value = value[:len(value)-1]
	case 'g', 'G':
		multiplier = 1 << 30
		value = value[:len(value)-1]
	}
	value = strings.TrimSpace(value)
	number, err := strconv.ParseUint(value, 10, 64)
	if err != nil || number == 0 {
		return 0, fmt.Errorf("gguf-split: invalid max size %q", original)
	}
	if number > math.MaxUint64/multiplier {
		return 0, errors.New("gguf-split: max size overflows uint64")
	}
	return number * multiplier, nil
}
