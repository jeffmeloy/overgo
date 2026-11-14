package landing

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
)

// bytePin is one record's hold on a file's current bytes.
type bytePin struct{ digest, record string }

// bytePins indexes the records that pin current bytes, each by its own field:
// a coverage document's inputs, the harness the capacity bundle was measured
// with, and the document digests validate's acceptance tests pass as literal
// arguments. Every other recorded digest is history.
func bytePins(repo string) (map[string][]bytePin, error) {
	pins := map[string][]bytePin{}
	coverage, err := filepath.Glob(filepath.Join(repo, "docs", "verification", "*-coverage.json"))
	if err != nil {
		return nil, err
	}
	for _, path := range coverage {
		var document struct{ Inputs map[string]string }
		if err := decodeRecord(path, &document); err != nil {
			return nil, err
		}
		for input, digest := range document.Inputs {
			pins[input] = append(pins[input], bytePin{digest, relativeRecord(repo, path)})
		}
	}
	capacity := filepath.Join(repo, "docs", "image_video_capacity.json")
	var bundle struct {
		SourceFiles map[string]struct{ SHA256 string } `json:"source_files"`
	}
	if err := decodeRecord(capacity, &bundle); err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	for source, file := range bundle.SourceFiles {
		pins[source] = append(pins[source], bytePin{file.SHA256, relativeRecord(repo, capacity)})
	}
	tests, err := filepath.Glob(filepath.Join(repo, "cmd", "validate", "*_test.go"))
	if err != nil {
		return nil, err
	}
	for _, path := range tests {
		syntax, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			var document, digest string
			for _, argument := range call.Args {
				literal, ok := argument.(*ast.BasicLit)
				if !ok || literal.Kind != token.STRING {
					continue
				}
				value, _ := strconv.Unquote(literal.Value)
				if artifact.ValidHexDigest(value) {
					digest = value
				} else if strings.HasPrefix(value, "docs/") {
					document = value
				}
			}
			if document != "" && digest != "" {
				pins[document] = append(pins[document], bytePin{digest, relativeRecord(repo, path)})
			}
			return true
		})
	}
	return pins, nil
}

func decodeRecord(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func relativeRecord(repo, path string) string {
	relative, _ := filepath.Rel(repo, path)
	return filepath.ToSlash(relative)
}

// stalePins names each planned path whose bytes -- raw, or with CRLF line ends normalized as a checkout may write them
// -- no longer match a pin, with the record that pins it and the edit that clears it, and then each record such an edit
// changes whose own bytes another record pins, to the chain's end (pin-chain/do). The edits are made here on copies, so
// each names the digest its path holds once every edit printed is made, and the refusal prints the whole chain at once:
// an edit printed to a coverage document that a cmd/validate literal pins in turn was refused again on the next
// attempt, twice for cuda-synthetic on 2026-11-14.
func stalePins(repo string, paths []string) ([]string, error) {
	pins, err := bytePins(repo)
	if err != nil {
		return nil, err
	}
	edited := map[string][]byte{}
	read := func(path string) ([]byte, error) {
		if data, found := edited[path]; found {
			return data, nil
		}
		data, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(path)))
		if os.IsNotExist(err) {
			return nil, nil
		}
		return data, err
	}
	type edit struct{ path, from, to string }
	edits, found := map[*bytePin]*edit{}, []*bytePin{}
	planned := slices.Compact(slices.Sorted(slices.Values(paths)))
	held := 0
	for _, pinned := range pins {
		held += len(pinned)
	}
	// Each round reads the records the last one edited; a round past one per pin could only follow a cycle of pins.
	for round := 0; ; round++ {
		if round > held {
			return nil, fmt.Errorf("the records' pins form a cycle through %s", strings.Join(slices.Sorted(maps.Keys(edited)), ", "))
		}
		reach := slices.Clone(planned)
		for _, record := range slices.Sorted(maps.Keys(edited)) {
			if !slices.Contains(reach, record) {
				reach = append(reach, record)
			}
		}
		changed := false
		for _, path := range reach {
			data, err := read(path)
			if err != nil {
				return nil, err
			}
			raw, normalized := sha256.Sum256(data), sha256.Sum256(bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n")))
			now := hex.EncodeToString(normalized[:])
			for index := range pins[path] {
				pin := &pins[path][index]
				if pin.digest == hex.EncodeToString(raw[:]) || pin.digest == now {
					continue
				}
				record, err := read(pin.record)
				if err != nil {
					return nil, err
				}
				edited[pin.record] = bytes.ReplaceAll(record, []byte(pin.digest), []byte(now))
				if edits[pin] == nil {
					edits[pin] = &edit{path: path, from: pin.digest}
					found = append(found, pin)
				}
				edits[pin].to, pin.digest, changed = now, now, true
			}
		}
		if !changed {
			break
		}
	}
	var stale, chained []string
	for _, pin := range found {
		change := edits[pin]
		sed := "sed -i s/" + change.from + "/" + change.to + "/ " + pin.record
		if slices.Contains(planned, change.path) {
			stale = append(stale, change.path+" (pinned by "+pin.record+"; edit: "+sed+", or else git checkout HEAD -- "+change.path+
				", the first only for a change that keeps the behaviour the record pins)")
		} else {
			chained = append(chained, change.path+" (pinned by "+pin.record+", once the edit to it above is made; edit: "+sed+")")
		}
	}
	slices.Sort(stale)
	return append(stale, chained...), nil
}
