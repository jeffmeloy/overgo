// Command audio-inspect measures and publishes admission for local audio.
//
// WAV admits PCM8/16/24/32, float32/64, G.711 and matching-width extensible
// subtypes. Native FLAC admits 4..24-bit samples, checking frame CRCs and the
// stream's PCM checksum when present. Other containers fail explicitly.
// Channels and sample rate are retained; no resampling or model execution runs.
//
// Parquet uses the existing local byte-array reader: an explicit, unique leaf
// and non-null value count, not whole-dataset validation. The policy bounds
// individual encoded and decoded payloads, not Parquet container-reader memory.
// Corpus files must remain immutable throughout inspection. JSON result lines
// are followed by the measured honesty summary; any quarantine returns nonzero.
// The policy and expectation JSON under internal/dataset/testdata are pinned
// validation fixtures, not recommended deployment thresholds.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
)

func main() {
	clioptions.MainNamed("audio-inspect", func() error { return run(context.Background(), os.Args[1:], os.Stdout) })
}

func run(ctx context.Context, args []string, output io.Writer) error {
	flags := flag.NewFlagSet("audio-inspect", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repo := flags.String("repo", "", "OvergoDB root")
	input := flags.String("input", "", "read-only WAV, FLAC, or Parquet file")
	inputArtifact := flags.String("input-artifact", "", "registered file artifact, instead of -input")
	policyPath := flags.String("policy", "", "explicit AudioInspectionPolicy JSON")
	column := flags.String("column", "", "Parquet byte-array leaf; empty inspects the complete audio file")
	limit := flags.Int("limit", 0, "required number of non-null Parquet values to inspect")
	expectPath := flags.String("expect", "", "optional pinned container and decoded-output expectations")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *repo == "" || (*input == "") == (*inputArtifact == "") || *policyPath == "" ||
		(*column == "" && *limit != 0) || (*column != "" && *limit <= 0) {
		return errors.New("usage: audio-inspect -repo <store> (-input <file> | -input-artifact <id>) -policy <json> [-column <leaf> -limit <values>] [-expect <json>]")
	}
	policyFile, err := os.Open(*policyPath)
	if err != nil {
		return err
	}
	var policy dataset.AudioInspectionPolicy
	err = strictjson.DecodeBounded(policyFile, artifact.MaxContentBytes, &policy)
	_ = policyFile.Close()
	if err != nil {
		return err
	}
	if err := policy.Validate(); err != nil {
		return err
	}
	var expected *inspectionExpectations
	if *expectPath != "" {
		expectFile, err := os.Open(*expectPath)
		if err != nil {
			return err
		}
		expected = new(inspectionExpectations)
		err = strictjson.DecodeBounded(expectFile, artifact.MaxContentBytes, expected)
		_ = expectFile.Close()
		if err != nil {
			return err
		}
		count := *limit
		if *column == "" {
			count = 1
		}
		if len(expected.Observations) != count {
			return errors.New("audio-inspect: expectation count differs from requested inputs")
		}
	}
	store, err := overgodb.Open(*repo)
	if err != nil {
		return err
	}
	defer store.Close()
	inputPath := *input
	var requested artifact.ID
	if *inputArtifact != "" {
		requested, err = artifact.ParseID(*inputArtifact)
		if err != nil {
			return err
		}
		if requested.Kind() != artifact.KindFile {
			return errors.New("audio-inspect: input artifact must be a file")
		}
		inputPath, err = artifact.AvailablePath(ctx, store, requested, artifact.LocationFile)
		if err != nil {
			return err
		}
	}
	file, err := os.Open(inputPath)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New("audio-inspect: input must be a regular file")
	}
	if *column == "" && uint64(info.Size()) > policy.MaximumEncodedBytes {
		return errors.New("audio-inspect: input exceeds encoded payload budget")
	}
	id, size, err := artifact.Identify(artifact.KindFile, file)
	if err != nil {
		return err
	}
	if expected != nil && expected.ContainerSHA256 != id.DigestHex() {
		return errors.New("audio-inspect: container differs from pinned expectation")
	}
	if requested.Valid() && requested != id {
		return errors.New("audio-inspect: input bytes differ from registered artifact")
	}
	location, err := artifact.CanonicalLocalLocation(id, artifact.LocationFile, inputPath)
	if err != nil {
		return err
	}
	descriptor, found, err := store.Artifact(ctx, id)
	if err != nil {
		return err
	}
	if found && descriptor.Size != size {
		return errors.New("audio-inspect: source descriptor size differs")
	}
	locationID, err := artifact.JSONID(artifact.KindEvidence, location)
	if err != nil {
		return err
	}
	batch := artifact.Batch{Key: "audio/inspection-source/" + locationID.DigestHex(),
		Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}}}
	if !found {
		descriptor = artifact.Descriptor{ID: id, Size: size}
	}
	batch.Artifacts = []artifact.Descriptor{descriptor}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return err
	}
	encoder := json.NewEncoder(output)
	var inspected, accepted uint64
	observe := func(index uint64, data []byte) error {
		result, err := dataset.InspectAudio(ctx, store, data, dataset.AudioPayloadOrigin{
			Container: id, Column: *column, ValueIndex: index,
		}, policy)
		if err != nil {
			return err
		}
		if expected != nil {
			want := expected.Observations[index]
			if result.Signal.Source.Audio.DigestHex() != want.EncodedSHA256 || result.DecodedSHA256 != want.DecodedSHA256 ||
				result.Signal.FrameCount != want.Frames || result.Signal.Format != want.Format || result.Decision.Outcome != want.Outcome {
				return fmt.Errorf("audio-inspect: decoded output differs from expectation at value %d", index)
			}
		}
		inspected++
		if result.Decision.Outcome == recipecontract.AudioAdmissionAccepted {
			accepted++
		}
		return encoder.Encode(result)
	}
	if *column != "" {
		err = dataset.ReadParquetTextRows(location.Value, *column, *limit, func(index uint64, value string) error {
			return observe(index, []byte(value))
		})
	} else {
		if _, err := file.Seek(0, io.SeekStart); err != nil {
			return err
		}
		// Include one excess byte to reject growth without an unbounded read.
		data, readErr := io.ReadAll(io.LimitReader(file, int64(policy.MaximumEncodedBytes)+1))
		if readErr != nil {
			return readErr
		}
		err = observe(0, data)
	}
	if err != nil {
		return err
	}
	var verified uint64
	if expected != nil {
		verified = inspected
	}
	if _, err := fmt.Fprintf(output, "audio-inspect: inspected=%d accepted=%d quarantined=%d verified=%d; decoded CPU samples only; no model inference, training, or GPU execution\n", inspected, accepted, inspected-accepted, verified); err != nil {
		return err
	}
	if *column != "" && inspected != uint64(*limit) {
		return errors.New("audio-inspect: container has fewer non-null values than requested")
	}
	if accepted != inspected {
		return errors.New("audio-inspect: quarantined input")
	}
	return nil
}

type inspectionExpectations struct {
	ContainerSHA256 string `json:"container_sha256"`
	Oracle          string `json:"oracle"`
	Observations    []struct {
		EncodedSHA256 string                               `json:"encoded_sha256"`
		DecodedSHA256 string                               `json:"decoded_sha256"`
		Frames        uint64                               `json:"frames"`
		Format        recipecontract.AudioFormat           `json:"format"`
		Outcome       recipecontract.AudioAdmissionOutcome `json:"outcome"`
	} `json:"observations"`
}
