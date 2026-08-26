// code-manifest emits the canonical structural authority for a source tree.
package main

import (
	"flag"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/codemanifest"
	"overgo/internal/repoanalysis"
)

func main() {
	clioptions.MainNamed("code-manifest", run)
}

func run() error {
	root := flag.String("root", ".", "repository root")
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
