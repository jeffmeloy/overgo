// api-manifest compiles the repository's public API manifest: every
// command binary, every typed document contract the code declares, the
// UTCP-style tool protocol, and the versioned authority files that
// govern the gate. -update writes the canonical JSON document and a
// readable summary to docs/; -check verifies both are current.
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

const (
	manifestJSONPath     = "docs/api_manifest.json"
	manifestMarkdownPath = "docs/API_MANIFEST.md"
)

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
	update := flag.Bool("update", false, "write docs/api_manifest.json and docs/API_MANIFEST.md")
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
	markdown := renderMarkdown(manifest)
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
			return fmt.Errorf("api-manifest: %s is stale (have %s want %s); regenerate with -update",
				manifestJSONPath, parsed.ID, manifest.ID)
		}
		fmt.Printf("api-manifest: current at %s\n", manifest.ID)
		return nil
	}
	if err := clioptions.WriteOutputFile(filepath.FromSlash(manifestJSONPath), content.Data); err != nil {
		return err
	}
	if err := clioptions.WriteOutputFile(filepath.FromSlash(manifestMarkdownPath), []byte(markdown)); err != nil {
		return err
	}
	fmt.Printf("wrote %s and %s (%s)\n", manifestJSONPath, manifestMarkdownPath, manifest.ID)
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

func renderMarkdown(manifest apimanifest.Manifest) string {
	var out strings.Builder
	out.WriteString("# API manifest\n\n")
	out.WriteString("Generated by `go run ./cmd/api-manifest -update` from the source tree. Do not edit this file directly; the canonical document is [api_manifest.json](api_manifest.json), content-addressed as `" + manifest.ID.String() + "`.\n\n")
	fmt.Fprintf(&out, "Source identity `%s`; %d binaries, %d routes, %d document contracts, %d protocols, %d authorities.\n\n",
		manifest.SourceIdentity, len(manifest.Binaries), len(manifest.Routes), len(manifest.Documents), len(manifest.Protocols), len(manifest.Authorities))
	out.WriteString("## Binaries\n\n")
	out.WriteString("Every command under `cmd/`; invoke as `go run ./cmd/<name>` or build to `bin/`.\n\n")
	out.WriteString("| Binary | Package |\n| --- | --- |\n")
	for _, binary := range manifest.Binaries {
		fmt.Fprintf(&out, "| `%s` | `%s` |\n", binary.Name, binary.Package)
	}
	out.WriteString("\n## Routes\n\n")
	out.WriteString("Every public HTTP method and path projected from the runtime route catalog.\n\n")
	out.WriteString("| Method | Path | Authentication |\n| --- | --- | --- |\n")
	for _, route := range manifest.Routes {
		fmt.Fprintf(&out, "| %s | `%s` | %s |\n", route.Method, route.Path, route.Authentication)
	}
	out.WriteString("\n## Document contracts\n\n")
	out.WriteString("Every typed store document the code declares: kind, wire media type, schema, and the owning package.\n\n")
	out.WriteString("| Kind | Media type | Schema | Owner |\n| --- | --- | --- | --- |\n")
	for _, document := range manifest.Documents {
		fmt.Fprintf(&out, "| %s | `%s` | `%s` | `%s` |\n", document.Kind, document.MediaType, document.Schema, document.Owner)
	}
	out.WriteString("\n## Protocols\n\n")
	out.WriteString("| Protocol | Transport | Owner | Contract |\n| --- | --- | --- | --- |\n")
	for _, protocol := range manifest.Protocols {
		contract := ""
		if len(protocol.Contracts) > 0 {
			contract = "`" + protocol.Contracts[0].Schema + "`"
		}
		fmt.Fprintf(&out, "| %s | %s | `%s` | %s |\n", protocol.Name, protocol.Transport, protocol.Owner, contract)
	}
	out.WriteString("\nAgent tool calling is UTCP-style: each tool is described by a manual document (effect class, typed arguments, native transport binding) registered in OvergoDB; orchestration resolves tools from the store and invokes them over their native transports. There is no MCP bridge by design.\n")
	out.WriteString("\n## Authorities\n\n")
	out.WriteString("Versioned files that govern automation, named with content digests.\n\n")
	out.WriteString("| Authority | Path | SHA-256 |\n| --- | --- | --- |\n")
	for _, authority := range manifest.Authorities {
		fmt.Fprintf(&out, "| %s | [%s](../%s) | `%s` |\n", authority.Name, authority.Path, authority.Path, authority.ContentID[:16])
	}
	return out.String()
}
