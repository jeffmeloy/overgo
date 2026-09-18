package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// This source boundary isolates the reviewed IMRoPE correction. Historical
// media timings remain historical; only unchanged numerical paths are reused.
const mediaRotaryBase = "e6cd9abcb3fe9dcb52a638775064dae94a6266b3"
const mediaRotaryAfter = "408c978bebc76095a40aa919c0102c864ac0bb1f"

// Reviewed pairing patch; binds every source byte before this commit exists.
const mediaRotaryPairingDigest = "ee811131bb38c9eece3de4bd7ce47c3631fa495ab6596f241394931c9b6afd7a"

var mediaRotaryPaths = []string{
	"internal/cuda/executor/executor.go",
	"internal/cuda/executor/kernel_bindings_generated.go", "internal/cuda/executor/launch_rope.go",
	"internal/cuda/kernel/manifest_generated.go", "internal/cuda/kernel/ops_f32.ptx",
	"internal/tensor/builder_rope.go", "internal/tensor/reference/ops_rope.go", "internal/tensor/tensor.go",
	"kernels/cuda/ops_f32.cu", "kernels/manifest.json",
}

func mediaRotarySources(root, revision string) (map[string][]byte, error) {
	files := make(map[string][]byte, len(mediaRotaryPaths))
	for _, path := range mediaRotaryPaths {
		var data []byte
		var err error
		if revision == "" {
			data, err = os.ReadFile(filepath.Join(root, path))
		} else {
			command := exec.Command("git", "show", revision+":"+path)
			command.Dir = root
			data, err = command.Output()
		}
		if err != nil {
			return nil, err
		}
		files[path] = bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	}
	return files, nil
}

func mediaSyntax(node ast.Node) string {
	var out bytes.Buffer
	if err := printer.Fprint(&out, token.NewFileSet(), node); err != nil {
		panic(err)
	}
	return out.String()
}

func mediaRotaryGoBoundary(path string, data []byte) (string, error) {
	file, err := parser.ParseFile(token.NewFileSet(), path, data, parser.SkipObjectResolution)
	if err != nil {
		return "", err
	}
	var failure error
	ast.Inspect(file, func(node ast.Node) bool {
		switch value := node.(type) {
		case *ast.FuncDecl:
			if value.Name.Name == "RoPEWithOptions" && len(value.Body.List) > 0 {
				if branch, ok := value.Body.List[0].(*ast.IfStmt); ok && strings.Contains(mediaSyntax(branch.Cond), "MultiPositions") {
					if len(value.Body.List) < 2 {
						failure = errors.New("incomplete rotary guard")
						return false
					}
					second, ok := value.Body.List[1].(*ast.IfStmt)
					if !ok || mediaSyntax(branch.Cond) != "options.MultiPositions != nil" ||
						mediaSyntax(second.Cond) != "options.InterleavedSections || options.Sections != [MaxDimensions]int32{}" {
						failure = errors.New("single-axis rotary guard changed")
						return false
					}
					value.Body.List = value.Body.List[2:]
				}
			}
		case *ast.TypeSpec:
			if value.Name.Name == "RoPEOptions" {
				fields := value.Type.(*ast.StructType).Fields
				fields.List = slices.DeleteFunc(fields.List, func(field *ast.Field) bool {
					return len(field.Names) == 1 && slices.Contains([]string{"MultiPositions", "Sections", "InterleavedSections"}, field.Names[0].Name)
				})
			}
		case *ast.SwitchStmt:
			value.Body.List = slices.DeleteFunc(value.Body.List, func(statement ast.Stmt) bool {
				clause, ok := statement.(*ast.CaseClause)
				return ok && len(clause.List) == 1 && mediaSyntax(clause.List[0]) == "tensor.OpRoPEMulti"
			})
		case *ast.ValueSpec:
			if len(value.Names) != 1 || len(value.Values) != 1 {
				break
			}
			if slices.Contains([]string{"BundleABIVersion", "OpsF32SHA256"}, value.Names[0].Name) {
				value.Values[0] = &ast.BasicLit{Kind: token.INT, Value: "0"}
			}
			if value.Names[0].Name == "kernelFunctionArgumentCounts" {
				index := -1
				ast.Inspect(file, func(node ast.Node) bool {
					if names, ok := node.(*ast.ValueSpec); ok && len(names.Names) == 1 && names.Names[0].Name == "kernelFunctionNames" {
						for i, element := range names.Values[0].(*ast.CompositeLit).Elts {
							if mediaSyntax(element) == `"rope_multi_f32"` {
								index = i
							}
						}
					}
					return true
				})
				values := value.Values[0].(*ast.CompositeLit)
				if index < 0 || index >= len(values.Elts) {
					failure = errors.New("missing multi-axis kernel ABI")
					break
				}
				values.Elts[index] = &ast.BasicLit{Kind: token.INT, Value: "0"}
			}
		}
		return true
	})
	file.Decls = slices.DeleteFunc(file.Decls, func(declaration ast.Decl) bool {
		if function, ok := declaration.(*ast.FuncDecl); ok {
			return slices.Contains([]string{"RoPEMultiScaled", "buildRoPEMulti", "ropeMulti"}, function.Name.Name)
		}
		if group, ok := declaration.(*ast.GenDecl); ok && len(group.Specs) == 1 {
			if kind, ok := group.Specs[0].(*ast.TypeSpec); ok {
				return kind.Name.Name == "RoPEMultiAttributes"
			}
		}
		return false
	})
	return mediaSyntax(file), failure
}

func mediaWithoutKernel(data []byte, marker string) ([]byte, error) {
	if bytes.Count(data, []byte(marker)) != 1 {
		return nil, errors.New("kernel boundary is absent or ambiguous")
	}
	start := bytes.Index(data, []byte(marker))
	open := bytes.IndexByte(data[start:], '{') + start
	if open < start {
		return nil, errors.New("kernel body absent")
	}
	depth := 0
	for i := open; i < len(data); i++ {
		switch data[i] {
		case '{':
			depth++
		case '}':
			depth--
		}
		if depth == 0 {
			return append(slices.Clone(data[:start]), data[i+1:]...), nil
		}
	}
	return nil, errors.New("kernel body is incomplete")
}

func mediaRotaryManifest(data []byte) ([]byte, error) {
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, err
	}
	delete(document, "abiVersion")
	for _, item := range document["modules"].([]any) {
		module := item.(map[string]any)
		if module["source"] != "kernels/cuda/ops_f32.cu" {
			continue
		}
		delete(module, "sourceSha256")
		delete(module, "assetSha256")
		for _, item := range module["argumentLayouts"].([]any) {
			layout := item.(map[string]any)
			functions := layout["functions"].([]any)
			if len(functions) == 1 && functions[0] == "rope_multi_f32" {
				delete(layout, "parameters")
			}
		}
	}
	return json.Marshal(document)
}

func compareMediaRotarySources(before, after map[string][]byte) error {
	if len(before) != len(mediaRotaryPaths) || len(after) != len(mediaRotaryPaths) {
		return errors.New("incomplete rotary source boundary")
	}
	for _, path := range mediaRotaryPaths {
		left, right := before[path], after[path]
		if len(left) == 0 || len(right) == 0 {
			return fmt.Errorf("missing source %s", path)
		}
		var err error
		switch {
		case strings.HasSuffix(path, ".go"):
			var a, b string
			a, err = mediaRotaryGoBoundary(path, left)
			if err == nil {
				b, err = mediaRotaryGoBoundary(path, right)
			}
			left, right = []byte(a), []byte(b)
		case strings.HasSuffix(path, ".cu"), strings.HasSuffix(path, ".ptx"):
			marker := `extern "C" __global__ void rope_multi_f32(`
			if strings.HasSuffix(path, ".ptx") {
				marker = ".visible .entry rope_multi_f32("
			}
			left, err = mediaWithoutKernel(left, marker)
			if err == nil {
				right, err = mediaWithoutKernel(right, marker)
			}
		default:
			left, err = mediaRotaryManifest(left)
			if err == nil {
				right, err = mediaRotaryManifest(right)
			}
		}
		if err != nil {
			return fmt.Errorf("media numerical boundary changed in %s: %w", path, err)
		}
		if !bytes.Equal(left, right) {
			return fmt.Errorf("media numerical boundary changed in %s", path)
		}
	}
	return nil
}

func mediaSingleAxisSites(path string, data []byte) (int, error) {
	file, err := parser.ParseFile(token.NewFileSet(), path, data, parser.SkipObjectResolution)
	if err != nil {
		return 0, err
	}
	var failure error
	proved := map[*ast.SelectorExpr]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "RoPEWithOptions" {
			return true
		}
		if len(call.Args) != 2 {
			failure = errors.New("unresolved rotary arguments")
			return false
		}
		options, ok := call.Args[1].(*ast.CompositeLit)
		if !ok {
			failure = errors.New("unresolved rotary options")
			return false
		}
		for _, field := range options.Elts {
			pair, ok := field.(*ast.KeyValueExpr)
			if !ok || slices.Contains([]string{"MultiPositions", "Sections", "InterleavedSections"}, mediaSyntax(pair.Key)) {
				failure = errors.New("multi-axis options")
				return false
			}
		}
		proved[selector] = true
		return true
	})
	ast.Inspect(file, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok {
			if slices.Contains([]string{"RoPEMultiScaled", "OpRoPEMulti", "RoPEMultiAttributes"}, selector.Sel.Name) || selector.Sel.Name == "RoPEWithOptions" && !proved[selector] {
				failure = errors.New("unresolved or multi-axis rotary consumer")
			}
		}
		return true
	})
	return len(proved), failure
}

func checkMediaRotaryConsumers(root string, paths []string) error {
	command := exec.Command("git", append([]string{"ls-tree", "-r", "--name-only", mediaRotaryBase, "--"}, paths...)...)
	command.Dir = root
	raw, err := command.Output()
	if err != nil {
		return err
	}
	sites := 0
	for path := range strings.FieldsSeq(string(raw)) {
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") ||
			strings.HasPrefix(path, "internal/tensor/") || strings.HasPrefix(path, "internal/cuda/") {
			continue
		}
		command := exec.Command("git", "show", mediaRotaryBase+":"+path)
		command.Dir = root
		data, err := command.Output()
		if err != nil {
			return err
		}
		count, err := mediaSingleAxisSites(path, data)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		sites += count
	}
	if sites == 0 {
		return errors.New("empty rotary consumer proof")
	}
	return nil
}

func checkMediaRotarySource(root, revision string, paths []string) (string, error) {
	changed, err := mediaRuntimeChanges(root, mediaRotaryBase, revision, paths)
	if err != nil {
		return "", err
	}
	for _, path := range changed {
		if !slices.Contains(mediaRotaryPaths, path) {
			return "", fmt.Errorf("unreconciled media source %s", path)
		}
	}
	before, err := mediaRotarySources(root, mediaRotaryBase)
	if err != nil {
		return "", err
	}
	after, err := mediaRotarySources(root, revision)
	if err != nil {
		return "", err
	}
	// The masks below prove this exact reviewed patch, not a standing exception
	// for arbitrary future edits to rotary code, metadata, or generated assets.
	bound, err := mediaRotarySources(root, mediaRotaryAfter)
	if err != nil {
		return "", err
	}
	if err := matchMediaRotarySource(after, bound); err != nil {
		return "", err
	}
	if err := compareMediaRotarySources(before, after); err != nil {
		return "", err
	}
	if err := checkMediaRotaryConsumers(root, paths); err != nil {
		return "", err
	}
	return mediaRotaryBase, nil
}

func matchMediaRotarySource(observed, bound map[string][]byte) error {
	if mediaRotarySourceDigest(observed) == mediaRotaryPairingDigest {
		return nil
	}
	for _, path := range mediaRotaryPaths {
		if len(bound[path]) == 0 || !bytes.Equal(observed[path], bound[path]) {
			return fmt.Errorf("rotary proof source differs: %s", path)
		}
	}
	return nil
}

func mediaRotarySourceDigest(sources map[string][]byte) string {
	hash := sha256.New()
	for _, path := range mediaRotaryPaths {
		data := sources[path]
		if len(data) == 0 {
			return ""
		}
		fmt.Fprintf(hash, "%d:%s%d:", len(path), path, len(data))
		hash.Write(data)
	}
	return fmt.Sprintf("%x", hash.Sum(nil))
}

func TestMediaRotarySourceIsolation(t *testing.T) {
	for _, expression := range []string{
		"b.RoPEWithOptions(x, options)",
		"b.RoPEWithOptions(x, tensor.RoPEOptions{MultiPositions: axes})",
		"b.RoPEWithOptions", "tensor.OpRoPEMulti",
	} {
		if _, err := mediaSingleAxisSites("counterexample.go", []byte("package p; var value = "+expression)); err == nil {
			t.Fatal("unresolved consumer accepted", expression)
		}
	}
	root := testutil.RepoRoot(t)
	var merged mediaMergedEvidence
	raw, err := os.ReadFile(filepath.Join(root, "docs/image_video_merged.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &merged); err != nil {
		t.Fatal(err)
	}
	if _, err := checkMediaRotarySource(root, "", merged.RuntimePaths); err != nil {
		t.Fatal(err)
	}
	before, err := mediaRotarySources(root, mediaRotaryBase)
	if err != nil {
		t.Fatal(err)
	}
	after, err := mediaRotarySources(root, "")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range mediaRotaryPaths {
		t.Run(path, func(t *testing.T) {
			copy := maps.Clone(after)
			delete(copy, path)
			if compareMediaRotarySources(before, copy) == nil {
				t.Fatal("omitted source accepted")
			}
			// Includes generated ABI/version/hash values erased by the semantic
			// comparison: exact source binding must still reject their mutation.
			copy[path] = append(slices.Clone(after[path]), '\n')
			if matchMediaRotarySource(copy, after) == nil {
				t.Fatal("changed proof source accepted")
			}
		})
	}
	path := "internal/tensor/builder_rope.go"
	tampered := slices.Clone(after[path])
	after[path] = bytes.ReplaceAll(tampered, []byte("options.MultiPositions != nil"), []byte("options.MultiPositions == nil"))
	if compareMediaRotarySources(before, after) == nil {
		t.Fatal("reachable changed branch accepted")
	}
	after[path] = tampered
	path = "internal/cuda/kernel/ops_f32.ptx"
	after[path] = append([]byte(strconv.Quote("unrelated PTX change")), after[path]...)
	if compareMediaRotarySources(before, after) == nil {
		t.Fatal("unrelated compiled code change accepted")
	}
}
