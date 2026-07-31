package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"llamacpp2go/internal/gguf"
)

type metadataReport struct {
	Key         string `json:"key"`
	Type        string `json:"type"`
	ElementType string `json:"elementType,omitempty"`
	Count       int    `json:"count,omitempty"`
	Value       any    `json:"value,omitempty"`
	Truncated   bool   `json:"truncated,omitempty"`
}

type tensorReport struct {
	Name   string   `json:"name"`
	Type   string   `json:"type"`
	Shape  []uint64 `json:"shape"`
	Shard  uint16   `json:"shard"`
	Offset uint64   `json:"offset"`
	Size   uint64   `json:"size"`
}

type report struct {
	Path          string           `json:"path"`
	Version       uint32           `json:"version"`
	Alignment     uint64           `json:"alignment"`
	DataOffset    uint64           `json:"dataOffset"`
	DataSize      uint64           `json:"dataSize"`
	SplitCount    uint16           `json:"splitCount"`
	MetadataCount int              `json:"metadataCount"`
	TensorCount   int              `json:"tensorCount"`
	TypeCounts    map[string]int   `json:"typeCounts"`
	Metadata      []metadataReport `json:"metadata,omitempty"`
	Tensors       []tensorReport   `json:"tensors,omitempty"`
}

func buildReport(path string, file *gguf.File, includeMetadata, includeTensors bool, maxString int) report {
	result := report{
		Path:          path,
		Version:       file.Version,
		Alignment:     file.Alignment,
		DataOffset:    file.DataOffset,
		DataSize:      file.DataSize,
		SplitCount:    file.SplitCount,
		MetadataCount: len(file.Metadata),
		TensorCount:   len(file.Tensors),
		TypeCounts:    make(map[string]int),
	}
	for _, tensor := range file.Tensors {
		result.TypeCounts[tensor.Type.String()]++
		if includeTensors {
			shape := make([]uint64, tensor.Dimensions)
			copy(shape, tensor.Shape[:tensor.Dimensions])
			result.Tensors = append(result.Tensors, tensorReport{
				Name:   tensor.Name,
				Type:   tensor.Type.String(),
				Shape:  shape,
				Shard:  tensor.Shard,
				Offset: tensor.Offset,
				Size:   tensor.Size,
			})
		}
	}
	if includeMetadata {
		for _, item := range file.Metadata {
			entry := metadataReport{Key: item.Key, Type: item.Value.Type.String()}
			if item.Value.Type == gguf.ValueTypeArray {
				entry.ElementType = item.Value.ArrayType.String()
				entry.Count = item.Value.Count()
			} else {
				entry.Value = item.Value.Data
				if value, ok := entry.Value.(string); ok && maxString >= 0 {
					runes := []rune(value)
					if len(runes) > maxString {
						entry.Value = string(runes[:maxString])
						entry.Truncated = true
					}
				}
			}
			result.Metadata = append(result.Metadata, entry)
		}
	}
	return result
}

func run(arguments []string) error {
	flags := flag.NewFlagSet("inspect-gguf", flag.ContinueOnError)
	includeMetadata := flags.Bool("metadata", true, "include metadata keys and scalar values")
	includeTensors := flags.Bool("tensors", false, "include every tensor descriptor")
	maxString := flags.Int("max-string", 256, "maximum characters shown for scalar strings; negative means unlimited")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return errors.New("usage: inspect-gguf [options] <model.gguf>")
	}
	path := flags.Arg(0)
	file, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(buildReport(path, file, *includeMetadata, *includeTensors, *maxString))
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "inspect-gguf:", err)
		os.Exit(1)
	}
}
