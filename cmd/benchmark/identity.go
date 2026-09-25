package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"hash"
	"os"
	"runtime/debug"

	"overgo/internal/artifact"
	"overgo/internal/benchmarkrecord"
	"overgo/internal/tokenizer"
)

const tokenDigestDomain = "overgo/benchmark/token-ids/v1\x00"

type tokenDigest struct {
	hash  hash.Hash
	count int
}

func newTokenDigest() *tokenDigest {
	digest := &tokenDigest{hash: sha256.New()}
	_, _ = digest.hash.Write([]byte(tokenDigestDomain))
	return digest
}

func (digest *tokenDigest) add(id tokenizer.TokenID) {
	var encoded [4]byte
	binary.LittleEndian.PutUint32(encoded[:], uint32(id))
	_, _ = digest.hash.Write(encoded[:])
	digest.count++
}

func (digest *tokenDigest) sum() string {
	return hex.EncodeToString(digest.hash.Sum(nil))
}

// JSON framing prevents ambiguity between adjacent prompt strings.
func benchmarkWorkloadDigest(prompts []string) string {
	id, _ := artifact.JSONID(artifact.KindEvidence, prompts)
	return id.DigestHex()
}

func benchmarkPromptDigest(prompt string) string {
	id, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte(prompt))
	return id.DigestHex()
}

// The Go module/build settings bind the binary image; the source commit is
// recorded separately because development binaries can have different flags.
func benchmarkModuleDigest() (string, error) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "", errors.New("benchmark: executable build identity is unavailable")
	}
	sum := sha256.Sum256([]byte(info.String()))
	return hex.EncodeToString(sum[:]), nil
}

func benchmarkAdapterIDs(paths []string) ([]artifact.ID, error) {
	ids := make([]artifact.ID, len(paths))
	for index, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		ids[index], _, err = artifact.Identify(artifact.KindAdapter, file)
		closeErr := file.Close()
		if err != nil || closeErr != nil {
			return nil, errors.Join(err, closeErr)
		}
	}
	return ids, nil
}

func benchmarkConfiguration(options options, adapters []artifact.ID) benchmarkrecord.BenchmarkConfiguration {
	return benchmarkrecord.BenchmarkConfiguration{
		Tokens: options.Tokens, Runs: options.Runs, Warmup: options.Warmup,
		CachePrompt: options.CachePrompt, BatchSequences: options.BatchSequences,
		ContextShift: options.ContextShift, Speculative: options.Speculative,
		Temperature: options.Temperature, TopK: options.TopK, DeviceTopK: options.DeviceTopK,
		Adapters: adapters,
	}
}

func benchmarkOptionsDigest(options options, adapters []artifact.ID) (string, error) {
	return benchmarkConfiguration(options, adapters).Digest()
}

func validIdentityDigest(value string) bool {
	return artifact.ValidHexDigest(value)
}

func validHex(value string) bool {
	_, err := hex.DecodeString(value)
	return err == nil
}
