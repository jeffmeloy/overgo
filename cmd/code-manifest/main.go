// code-manifest emits the canonical structural authority for a source tree.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
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
		base, err := codemanifest.Parse(data)
		if err != nil {
			return err
		}
		delta, err := codemanifest.Diff(base, manifest)
		if err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(delta)
	}
	content, err := manifest.Content()
	if err != nil {
		return err
	}
	if _, err := os.Stdout.Write(content.Data); err != nil {
		return err
	}
	_, err = os.Stdout.Write([]byte{'\n'})
	return err
}

func readManifestFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.Size() <= 0 || uint64(info.Size()) > artifact.MaxContentBytes {
		return nil, errors.New("code-manifest: base document exceeds the content bound")
	}
	return os.ReadFile(path)
}
