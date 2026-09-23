package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
)

// surfaceRoots names the module-local package roots whose dependency closure is
// one modality's code surface. kernels marks a surface that the CUDA kernel
// sources also feed, so a kernel change moves it.
type surfaceRoots struct {
	roots []string
}

// modalitySurfaces maps each surface to the package roots whose closure pins its
// evidence. The go-list closure walk fans a shared dependency (a recipe, a
// kernel) out to exactly the surfaces that compile it, so a change is scoped to
// the surfaces it can actually move without any per-model knowledge.
var modalitySurfaces = map[SurfaceID]surfaceRoots{
	"inference":   {roots: []string{"./internal/inference", "./internal/cuda/executor", "./internal/modelrecipe"}},
	"evaluation":  {roots: []string{"./internal/inference", "./internal/cuda/executor", "./internal/modelrecipe", "./internal/evaluation"}},
	"image":       {roots: []string{"./internal/diffusionimage", "./internal/mediacapability", "./internal/latentimage", "./internal/oscillatorimage", "./internal/sensenovarecipe"}},
	"video":       {roots: []string{"./internal/latentvideo"}},
	"speech":      {roots: []string{"./internal/speechsynth", "./internal/speechrecognition", "./internal/audiodsp"}},
	"vqa":         {roots: []string{"./internal/vqaserve", "./internal/projector"}},
	"specialized": {roots: []string{"./internal/seq2seq", "./internal/seriesforecast", "./internal/tabularicl", "./internal/vitencoder"}},
	"training":    {roots: []string{"./internal/trainingworkflow", "./internal/adaptertrain", "./internal/densecausal", "./internal/hostmath"}},
}

// kindValidation lists the appropriate validations for one declared task, each
// bound to the surface that pins its evidence. Dispatch is by task, never by
// model name: a text inference model earns a guard and the accuracy benchmarks,
// an image-gen model its generation proof, and so on.
func kindValidation(task recipe.Task) []struct {
	Validation string
	Surface    SurfaceID
} {
	type cell = struct {
		Validation string
		Surface    SurfaceID
	}
	switch string(task) {
	case "inference":
		return []cell{{"guard", "inference"}, {"mmlu-pro", "evaluation"}, {"bbh", "evaluation"}, {"ifeval", "evaluation"}}
	case "projection":
		return []cell{{"projection-proof", "evaluation"}}
	case "image-gen":
		return []cell{{"image-proof", "image"}}
	case "video-gen":
		return []cell{{"video-proof", "video"}}
	case "speech", "transcription":
		return []cell{{"speech-proof", "speech"}}
	case "vqa":
		return []cell{{"vqa-proof", "vqa"}}
	case "seq2seq", "forecast", "tabular", "image-embedding":
		return []cell{{"specialized-proof", "specialized"}}
	case "training":
		return []cell{{"training-proof", "training"}}
	default:
		return nil
	}
}

// globalAuthorityPath reports a change that can move every surface: the module
// graph, any kernel source, or the CUDA driver/runtime. Mirrors the inference
// impact rule so a kernel or module change re-validates everything.
func globalAuthorityPath(path string) bool {
	slash := filepath.ToSlash(path)
	if slash == "go.mod" || slash == "go.sum" {
		return true
	}
	return strings.HasPrefix(slash, "kernels/") || strings.HasPrefix(slash, "internal/cuda/")
}

// testOrDocPath reports a change that moves no code surface: a Go test file or a
// documentation/spec file outside any package's compiled sources.
func testOrDocPath(path string) bool {
	slash := filepath.ToSlash(path)
	if strings.HasSuffix(slash, "_test.go") {
		return true
	}
	return strings.HasPrefix(slash, "docs/")
}

// surfaceAffected derives, from a commit's changed paths, which modality
// surfaces the change can move. A module/kernel/driver change moves every
// surface; a change under a surface's package closure moves that surface; test
// and documentation changes move nothing. This is the commit-change-driven core
// of "no unnecessary testing".
func surfaceAffected(ctx context.Context, root string, changed []string) (map[SurfaceID]AffectRecord, error) {
	global := ""
	for _, path := range changed {
		if testOrDocPath(path) {
			continue
		}
		if globalAuthorityPath(path) {
			global = filepath.ToSlash(path)
			break
		}
	}
	out := make(map[SurfaceID]AffectRecord, len(modalitySurfaces))
	for id, surface := range modalitySurfaces {
		if global != "" {
			out[id] = AffectRecord{Affected: true, Reason: "authority path " + global + " moves every surface"}
			continue
		}
		dirs, err := surfaceDirectories(ctx, root, surface.roots)
		if err != nil {
			return nil, err
		}
		rec := AffectRecord{}
		for _, path := range changed {
			if testOrDocPath(path) {
				continue
			}
			if dir, ok := containingDirectory(dirs, filepath.ToSlash(path)); ok {
				rec = AffectRecord{Affected: true, Reason: "changed " + filepath.ToSlash(path) + " in surface package " + dir}
				break
			}
		}
		out[id] = rec
	}
	return out, nil
}

// containingDirectory reports the surface package directory that holds a path.
func containingDirectory(dirs []string, path string) (string, bool) {
	for _, dir := range dirs {
		if path == dir || strings.HasPrefix(path, dir+"/") {
			return dir, true
		}
	}
	return "", false
}

// surfaceDirectories lists the module-local package directories, relative to
// root and slash-separated, in the dependency closure of the given roots. It
// follows the Go tool's own resolution so the surface tracks the imports.
func surfaceDirectories(ctx context.Context, root string, roots []string) ([]string, error) {
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	arguments := append([]string{"list", "-deps", "-json"}, roots...)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: arguments, Dir: absolute, Env: os.Environ(), Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, fmt.Errorf("validate: go list for %v: %w: %s", roots, err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("validate: go list for %v: exit=%d: %s", roots, receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	var dirs []string
	decoder := json.NewDecoder(&stdout)
	for {
		var pkg struct{ Dir string }
		if err := decoder.Decode(&pkg); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}
		relative, err := filepath.Rel(absolute, pkg.Dir)
		if err != nil || strings.HasPrefix(relative, "..") {
			continue // outside the module: a standard-library or cached dependency
		}
		dirs = append(dirs, filepath.ToSlash(relative))
	}
	slices.Sort(dirs)
	return slices.Compact(dirs), nil
}
