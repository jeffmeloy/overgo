// api-manifest compiles commands, routes, document contracts, protocols and
// authority digests into docs/api_manifest.json. -check verifies its identity.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/apimanifest"
	"overgo/internal/clioptions"
	"overgo/internal/codemanifest"
	"overgo/internal/repoanalysis"
	"overgo/internal/server"
)

const manifestJSONPath = "docs/api_manifest.json"

// authorityFiles are the versioned documents that govern automation;
// each is named in the manifest with its content digest.
var authorityFiles = []struct{ name, path string }{
	{"staged-surface", "docs/staged_surface.json"},
	{"build-retained", "build/RETAINED.json"},
	{"generation-sources", "internal/modelrecipe/generation_source_profiles.json"},
}

func main() {
	clioptions.MainNamed("api-manifest", run)
}

func run() error {
	update := flag.Bool("update", false, "write docs/api_manifest.json")
	check := flag.Bool("check", false, "verify the written manifest matches the source")
	flag.Parse()
	if *update == *check {
		return fmt.Errorf("api-manifest: exactly one of -update or -check")
	}
	manifest, err := compile(".")
	if err != nil {
		return err
	}
	content, err := manifest.Content()
	if err != nil {
		return err
	}
	if *check {
		previous, err := os.ReadFile(filepath.FromSlash(manifestJSONPath))
		if err != nil {
			return fmt.Errorf("api-manifest: %s is absent; regenerate with -update", manifestJSONPath)
		}
		parsed, err := apimanifest.Parse(previous)
		if err != nil {
			return err
		}
		if parsed.ID != manifest.ID {
			// Name what moved, not just that something did: the structured
			// diff turns a staleness refusal into an actionable review.
			changes, compareErr := apimanifest.Compare(parsed, manifest)
			if compareErr != nil {
				return fmt.Errorf("api-manifest: %s is stale and undiffable (%v); regenerate with -update",
					manifestJSONPath, compareErr)
			}
			for _, change := range changes {
				fmt.Printf("api-manifest: changed %s %s %s\n", change.Class, change.Kind, change.Key)
			}
			return fmt.Errorf("api-manifest: %s is stale (have %s want %s, %d change(s)); regenerate with -update",
				manifestJSONPath, parsed.ID, manifest.ID, len(changes))
		}
		fmt.Printf("api-manifest: current at %s\n", manifest.ID)
		return nil
	}
	if err := clioptions.WriteOutputFile(filepath.FromSlash(manifestJSONPath), content.Data); err != nil {
		return err
	}
	fmt.Printf("wrote %s (%s)\n", manifestJSONPath, manifest.ID)
	return nil
}

func compile(root string) (apimanifest.Manifest, error) {
	snapshot, err := repoanalysis.DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return apimanifest.Manifest{}, err
	}
	selection, err := repoanalysis.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return apimanifest.Manifest{}, err
	}
	declarations, err := codemanifest.DocumentDeclarations(snapshot, []repoanalysis.BuildSelection{selection})
	if err != nil {
		return apimanifest.Manifest{}, err
	}
	documents, err := apimanifest.CompileDocuments(declarations)
	if err != nil {
		return apimanifest.Manifest{}, err
	}
	binaries, err := commandBinaries(root, selection.Context)
	if err != nil {
		return apimanifest.Manifest{}, err
	}
	authorities, err := authorityDigests(root)
	if err != nil {
		return apimanifest.Manifest{}, err
	}
	var protocols []apimanifest.Protocol
	for _, document := range documents {
		if document.Schema == "overgo/agent-tool-manual/v1" {
			protocols = append(protocols, apimanifest.Protocol{
				Name: "agent-tool-manuals", Owner: document.Owner, Transport: "utcp-native",
				Contracts: []apimanifest.ContractRef{{Kind: document.Kind, MediaType: document.MediaType, Schema: document.Schema}},
			})
		}
	}
	return apimanifest.New(apimanifest.Manifest{
		Version:        1,
		Release:        "master",
		SourceIdentity: snapshot.Identity(),
		BuildContexts: []apimanifest.BuildContext{{
			ID: selection.Context, GOOS: hostPart(selection.Context, 0), GOARCH: hostPart(selection.Context, 1),
		}},
		Binaries:    binaries,
		Routes:      server.APIManifestRoutes(),
		Documents:   documents,
		Protocols:   protocols,
		Authorities: authorities,
	})
}

func hostPart(context string, index int) string {
	parts := strings.Split(context, "/")
	if index < len(parts) {
		return parts[index]
	}
	return context
}

func commandBinaries(root, context string) ([]apimanifest.Binary, error) {
	entries, err := os.ReadDir(filepath.Join(root, "cmd"))
	if err != nil {
		return nil, err
	}
	binaries := make([]apimanifest.Binary, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		binaries = append(binaries, apimanifest.Binary{
			Name:          entry.Name(),
			Package:       "overgo/cmd/" + entry.Name(),
			BuildContexts: []string{context},
		})
	}
	sort.Slice(binaries, func(i, j int) bool { return binaries[i].Name < binaries[j].Name })
	return binaries, nil
}

func authorityDigests(root string) ([]apimanifest.Authority, error) {
	authorities := make([]apimanifest.Authority, 0, len(authorityFiles))
	for _, file := range authorityFiles {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(file.path)))
		if err != nil {
			return nil, fmt.Errorf("api-manifest: authority %s: %w", file.name, err)
		}
		digest := sha256.Sum256(data)
		authorities = append(authorities, apimanifest.Authority{
			Name: file.name, Path: file.path, ContentID: hex.EncodeToString(digest[:]),
		})
	}
	return authorities, nil
}
