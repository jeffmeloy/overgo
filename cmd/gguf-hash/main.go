package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"llamacpp2go/internal/gguf"
)

const (
	exitSuccess              = 0
	exitFailure              = 1
	exitManifestMissingEntry = 3
	exitManifestUnknownHash  = 4
	exitManifestFileError    = 5
	manifestScannerBuffer    = 4096
	maxManifestLineBytes     = 1 << 20
)

type parameters struct {
	input    string
	xxh64    bool
	sha1     bool
	sha256   bool
	uuid     bool
	noLayer  bool
	manifest string
}

type manifestData struct {
	entries    map[string]string
	algorithms map[string]bool
}

func main() {
	var params parameters
	all := flag.Bool("all", false, "use xxh64, sha1, and sha256")
	flag.BoolVar(&params.xxh64, "xxh64", false, "use xxh64")
	flag.BoolVar(&params.sha1, "sha1", false, "use sha1")
	flag.BoolVar(&params.sha256, "sha256", false, "use sha256")
	flag.BoolVar(&params.uuid, "uuid", false, "generate llama.cpp UUIDv5")
	flag.BoolVar(&params.noLayer, "no-layer", false, "exclude per-tensor hashes")
	flag.StringVar(&params.manifest, "check", "", "verify against a manifest")
	flag.StringVar(&params.manifest, "c", "", "verify against a manifest")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-hash [options] input.gguf")
		os.Exit(2)
	}
	params.input = flag.Arg(0)
	if *all {
		params.xxh64 = true
		params.sha1 = true
		params.sha256 = true
	}
	code, err := execute(params, os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
	if code != 0 {
		os.Exit(code)
	}
}

func execute(params parameters, output io.Writer) (int, error) {
	var manifest manifestData
	if params.manifest != "" {
		var err error
		manifest, err = readManifest(params.manifest)
		if err != nil {
			return exitManifestFileError, err
		}
		if len(manifest.algorithms) == 0 {
			return exitManifestUnknownHash, fmt.Errorf(
				"gguf-hash: manifest %q has no known hash format",
				params.manifest,
			)
		}
		if !params.xxh64 && !params.sha1 && !params.sha256 && !params.uuid {
			switch {
			case manifest.algorithms["sha256"]:
				params.sha256 = true
			case manifest.algorithms["sha1"]:
				params.sha1 = true
			case manifest.algorithms["xxh64"]:
				params.xxh64 = true
			case manifest.algorithms["uuid"]:
				params.uuid = true
			}
		}
		fmt.Fprintf(output, "manifest  %s", params.manifest)
		for _, algorithm := range []string{"sha256", "sha1", "xxh64", "uuid"} {
			if manifest.algorithms[algorithm] {
				fmt.Fprintf(output, "  %s", algorithm)
			}
		}
		fmt.Fprintln(output)
	}
	if !params.xxh64 && !params.sha1 && !params.sha256 && !params.uuid {
		params.xxh64 = true
	}

	file, err := gguf.Open(params.input)
	if err != nil {
		return exitFailure, fmt.Errorf("gguf-hash: open input: %w", err)
	}
	defer file.Close()
	result, err := file.Hash(gguf.HashOptions{
		XXH64:     params.xxh64,
		SHA1:      params.sha1,
		SHA256:    params.sha256,
		UUID:      params.uuid,
		PerTensor: !params.noLayer,
	})
	if err != nil {
		return exitFailure, fmt.Errorf("gguf-hash: %w", err)
	}

	verification := verificationState{}
	if !params.noLayer {
		for _, tensor := range result.Tensors {
			target := params.input + ":" + tensor.Name
			for _, record := range records(tensor.Values, false) {
				printRecord(output, record, target, manifest, &verification, true)
			}
		}
	}
	for _, record := range records(result.Model, params.uuid) {
		printRecord(output, record, params.input, manifest, &verification, false)
	}
	if params.manifest == "" {
		return exitSuccess, nil
	}
	code := verification.exitCode()
	fmt.Fprintf(
		output,
		"\nVerification results for %s - %s\n",
		params.manifest,
		exitCodeName(code),
	)
	return code, nil
}

type digestRecord struct {
	algorithm string
	digest    string
}

func records(values gguf.HashValues, includeUUID bool) []digestRecord {
	result := make([]digestRecord, 0, 4)
	if values.XXH64 != "" {
		result = append(result, digestRecord{"xxh64", values.XXH64})
	}
	if values.SHA1 != "" {
		result = append(result, digestRecord{"sha1", values.SHA1})
	}
	if values.SHA256 != "" {
		result = append(result, digestRecord{"sha256", values.SHA256})
	}
	if includeUUID && values.UUID != "" {
		result = append(result, digestRecord{"uuid", values.UUID})
	}
	return result
}

type verificationState struct {
	tensorFound    bool
	tensorMismatch bool
	modelFound     bool
	modelMismatch  bool
}

func printRecord(
	output io.Writer,
	record digestRecord,
	target string,
	manifest manifestData,
	verification *verificationState,
	tensor bool,
) {
	if manifest.entries == nil {
		fmt.Fprintf(output, "%-8s  %s  %s\n", record.algorithm, record.digest, target)
		return
	}
	want, found := manifest.entries[manifestKey(record.algorithm, target)]
	status := "Not Found"
	if found {
		status = "Ok"
		if want != record.digest {
			status = "Mismatch"
		}
	}
	if tensor {
		verification.tensorFound = verification.tensorFound || found
		verification.tensorMismatch = verification.tensorMismatch || found && want != record.digest
	} else {
		verification.modelFound = verification.modelFound || found
		verification.modelMismatch = verification.modelMismatch || found && want != record.digest
	}
	fmt.Fprintf(
		output,
		"%-8s  %s  %s  -  %s\n",
		record.algorithm,
		record.digest,
		target,
		status,
	)
}

func (v verificationState) exitCode() int {
	if !v.modelFound {
		if !v.tensorFound {
			return exitManifestMissingEntry
		}
		if v.tensorMismatch {
			return exitFailure
		}
		return exitSuccess
	}
	if v.modelMismatch || v.tensorMismatch {
		return exitFailure
	}
	return exitSuccess
}

func readManifest(path string) (manifestData, error) {
	file, err := os.Open(path)
	if err != nil {
		return manifestData{}, fmt.Errorf("gguf-hash: open manifest: %w", err)
	}
	defer file.Close()
	result := manifestData{
		entries:    make(map[string]string),
		algorithms: make(map[string]bool),
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, manifestScannerBuffer), maxManifestLineBytes)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 3 {
			continue
		}
		algorithm := fields[0]
		switch algorithm {
		case "xxh64", "sha1", "sha256", "uuid":
		default:
			continue
		}
		result.algorithms[algorithm] = true
		result.entries[manifestKey(algorithm, fields[2])] = fields[1]
	}
	if err := scanner.Err(); err != nil {
		return manifestData{}, err
	}
	return result, nil
}

func manifestKey(algorithm, target string) string {
	return algorithm + "\x00" + target
}

func exitCodeName(code int) string {
	switch code {
	case exitSuccess:
		return "Success"
	case exitFailure:
		return "Failure"
	case exitManifestMissingEntry:
		return "Manifest Missing Entry"
	case exitManifestUnknownHash:
		return "Manifest Unknown Hash"
	case exitManifestFileError:
		return "Manifest File Error"
	default:
		return "Failure"
	}
}
