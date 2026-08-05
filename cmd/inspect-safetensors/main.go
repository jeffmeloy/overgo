package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"os"

	"llamacpp2go/internal/clioptions"
	"llamacpp2go/internal/hfgguf"
	"llamacpp2go/internal/hfrepo"
)

type tensorReport struct {
	Name     string   `json:"name"`
	Type     string   `json:"type"`
	Shape    []uint64 `json:"shape"`
	Shard    string   `json:"shard"`
	Elements uint64   `json:"elements"`
	Size     int64    `json:"size"`
}

type report struct {
	Path           string         `json:"path"`
	ModelType      string         `json:"modelType,omitempty"`
	TextModelType  string         `json:"textModelType,omitempty"`
	Architectures  []string       `json:"architectures,omitempty"`
	Companions     []string       `json:"companions,omitempty"`
	Shards         []string       `json:"shards"`
	TensorCount    int            `json:"tensorCount"`
	ParameterCount uint64         `json:"parameterCount"`
	DataSize       uint64         `json:"dataSize"`
	TypeCounts     map[string]int `json:"typeCounts"`
	Tensors        []tensorReport `json:"tensors,omitempty"`
	RuntimeCatalog *runtimeReport `json:"runtimeCatalog,omitempty"`
}

type runtimeReport struct {
	Architecture    string `json:"architecture"`
	BlockCount      uint32 `json:"blockCount"`
	EmbeddingLength uint32 `json:"embeddingLength"`
	VocabularySize  uint32 `json:"vocabularySize"`
}

func buildReport(repository *hfrepo.Repository, includeTensors, validateRuntime bool) (report, error) {
	result := report{
		Path:          repository.Directory,
		ModelType:     repository.Identity.ModelType,
		TextModelType: repository.Identity.TextModelType,
		Architectures: repository.Identity.Architectures,
		Companions:    repository.Companions,
		Shards:        repository.Tensors.Shards(),
		TensorCount:   len(repository.Tensors.Tensors),
		TypeCounts:    make(map[string]int),
	}
	for _, name := range repository.Tensors.Names() {
		tensor := repository.Tensors.Tensors[name]
		elements := tensor.Elements()
		if result.ParameterCount > math.MaxUint64-elements || tensor.Size() < 0 || result.DataSize > math.MaxUint64-uint64(tensor.Size()) {
			return report{}, errors.New("inspect-safetensors: repository totals overflow")
		}
		result.ParameterCount += elements
		result.DataSize += uint64(tensor.Size())
		result.TypeCounts[tensor.DType]++
		if includeTensors {
			result.Tensors = append(result.Tensors, tensorReport{
				Name: tensor.Name, Type: tensor.DType, Shape: tensor.Shape,
				Shard: tensor.Shard, Elements: elements, Size: tensor.Size(),
			})
		}
	}
	if validateRuntime {
		spec, err := hfgguf.ValidateRepository(repository)
		if err != nil {
			return report{}, err
		}
		result.RuntimeCatalog = &runtimeReport{
			Architecture: spec.Architecture, BlockCount: spec.BlockCount,
			EmbeddingLength: spec.EmbeddingLength, VocabularySize: spec.VocabularySize,
		}
	}
	return result, nil
}

func run(arguments []string) error {
	flags := flag.NewFlagSet("inspect-safetensors", flag.ContinueOnError)
	includeTensors := flags.Bool("tensors", false, "include every tensor descriptor")
	validateRuntime := flags.Bool("validate-runtime", false, "validate supported metadata and weights through the GGUF runtime catalogs")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return errors.New("usage: inspect-safetensors [options] <model-directory>")
	}
	repository, err := hfrepo.Open(flags.Arg(0))
	if err != nil {
		return err
	}
	defer repository.Close()
	result, err := buildReport(repository, *includeTensors, *validateRuntime)
	if err != nil {
		return err
	}
	return clioptions.WritePrettyJSON(os.Stdout, result)
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "inspect-safetensors:", err)
		os.Exit(1)
	}
}
