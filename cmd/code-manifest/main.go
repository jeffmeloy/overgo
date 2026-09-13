// code-manifest emits the canonical structural authority for a source tree.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"

	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/codemanifest"
	"overgo/internal/repoanalysis"
)

func main() {
	clioptions.MainNamed("code-manifest", run)
}

func run() error {
	root := flag.String("root", ".", "repository root")
	basePath := flag.String("base", "", "canonical base manifest to compare")
	closure := flag.Bool("closure", false, "with -base: emit reverse-reachable impact instead of the raw delta")
	ownershipSurface := flag.Bool("ownership-surface", false, "with -base and -closure: emit the automation ownership surface")
	flag.Parse()
	resolved, err := filepath.Abs(*root)
	if err != nil {
		return err
	}
	snapshot, err := repoanalysis.DiscoverGo(resolved, "internal", "cmd")
	if err != nil {
		return err
	}
	selection, err := repoanalysis.HostBuildSelection(resolved, "./internal/...", "./cmd/...")
	if err != nil {
		return err
	}
	manifest, err := codemanifest.Generate(snapshot, []repoanalysis.BuildSelection{selection}, nil)
	if err != nil {
		return err
	}
	if *basePath != "" {
		data, err := readManifestFile(*basePath)
		if err != nil {
			return err
		}
		base, err := codemanifest.Parse(bytes.TrimSuffix(data, []byte{'\n'}))
		if err != nil {
			return err
		}
		delta, err := codemanifest.Diff(base, manifest)
		if err != nil {
			return err
		}
		if !*closure {
			return json.NewEncoder(os.Stdout).Encode(delta)
		}
		impact, err := codemanifest.Close(base, manifest, delta)
		if err != nil {
			return err
		}
		if *ownershipSurface {
			return json.NewEncoder(os.Stdout).Encode(automationcheck.ManifestSurface(impact))
		}
		return json.NewEncoder(os.Stdout).Encode(impact)
	}
	return json.NewEncoder(os.Stdout).Encode(manifest)
}

func readManifestFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 {
		return nil, errors.New("code-manifest: base must be a nonempty regular analysis file")
	}
	return os.ReadFile(path)
}
