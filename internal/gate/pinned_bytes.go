package gate

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// bytePin is one record's hold on a file's current bytes.
type bytePin struct{ digest, record string }

// bytePins indexes the records that pin current bytes, each by its own field:
// a coverage document's inputs, the harness the capacity bundle was measured
// with, and the document digests the complete-mode compatibility tests pass
// as literal arguments. Every other recorded digest is history.
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
	tests, err := filepath.Glob(filepath.Join(repo, "cmd", "compatibility", "*_test.go"))
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
				if _, err := hex.DecodeString(value); err == nil && len(value) == sha256.Size*2 {
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

// stalePins names each planned path whose bytes -- raw, or with CRLF line
// ends normalized as a checkout may write them -- no longer match a pin,
// with the record that pins it.
func stalePins(repo string, paths []string) ([]string, error) {
	pins, err := bytePins(repo)
	if err != nil {
		return nil, err
	}
	var stale []string
	for _, path := range paths {
		held := pins[path]
		if len(held) == 0 {
			continue
		}
		data, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(path)))
		if err != nil && !os.IsNotExist(err) {
			return nil, err
		}
		raw, normalized := sha256.Sum256(data), sha256.Sum256(bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n")))
		for _, pin := range held {
			if pin.digest != hex.EncodeToString(raw[:]) && pin.digest != hex.EncodeToString(normalized[:]) {
				stale = append(stale, path+" (pinned by "+pin.record+")")
			}
		}
	}
	slices.Sort(stale)
	return stale, nil
}
