package codemanifest

import (
	"errors"
	"fmt"
	"path"
	"slices"
	"strings"

	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
)

const (
	analyzerName    = "overgo-codeprofile"
	analyzerVersion = "consumer-graph-v1"
)

type contextGraph struct {
	context      BuildContext
	selection    repoanalysis.BuildSelection
	declarations []codeprofile.ConsumerDeclaration
	references   []codeprofile.ConsumerReference
}

// Generate composes a canonical manifest from the repository's existing
// source snapshot, function profile, build selections, consumer graphs, and
// caller-declared content-identified external inputs.
func Generate(snapshot repoanalysis.SourceSnapshot, selections []repoanalysis.BuildSelection, external []ExternalInput) (Manifest, error) {
	if len(selections) == 0 || len(selections) > maxBuildContexts {
		return Manifest{}, errors.New("code manifest: invalid build selection count")
	}
	profile, err := codeprofile.Build(snapshot)
	if err != nil {
		return Manifest{}, fmt.Errorf("code manifest: profile: %w", err)
	}
	graphs := make([]contextGraph, 0, len(selections))
	seenContexts := map[string]bool{}
	for _, selection := range selections {
		context, err := buildContext(selection)
		if err != nil {
			return Manifest{}, err
		}
		if seenContexts[context.ID] {
			return Manifest{}, fmt.Errorf("code manifest: duplicate build context %q", context.ID)
		}
		seenContexts[context.ID] = true
		declarations, references, _, err := codeprofile.ProductionConsumerGraph(snapshot, selection)
		if err != nil {
			return Manifest{}, fmt.Errorf("code manifest: consumer graph for %s: %w", context.ID, err)
		}
		graphs = append(graphs, contextGraph{context: context, selection: selection, declarations: declarations, references: references})
	}
	manifest := Manifest{
		Version: Version, SourceIdentity: snapshot.Identity(),
		Analyzer:       Analyzer{Name: analyzerName, Version: analyzerVersion},
		ExternalInputs: slices.Clone(external),
	}
	for _, graph := range graphs {
		manifest.BuildContexts = append(manifest.BuildContexts, graph.context)
	}
	for _, source := range snapshot.Files {
		file, boundaries, err := manifestFile(source, graphs)
		if err != nil {
			return Manifest{}, err
		}
		manifest.Files = append(manifest.Files, file)
		manifest.Uncertainty = append(manifest.Uncertainty, boundaries...)
	}
	fingerprints := profileFingerprints(profile)
	for _, graph := range graphs {
		known := map[string]SymbolID{}
		for _, declaration := range graph.declarations {
			if !graph.selection.Files[declaration.File] {
				continue
			}
			kind, ok := functionKind(declaration.Kind)
			if !ok {
				continue
			}
			key := declaration.File + "\x00" + declaration.Receiver + "\x00" + declaration.Name
			facts, found := fingerprints[key]
			id := SymbolID{
				Package: path.Dir(declaration.File), Context: graph.context.ID,
				Receiver: declaration.Receiver, Name: declaration.Name, Kind: kind,
			}
			if !found {
				manifest.Uncertainty = append(manifest.Uncertainty, Uncertainty{
					Kind: UncertaintyAnalysis, Path: declaration.File, Context: graph.context.ID, Symbol: &id,
					Reason: "function declaration has no profiled body and signature",
				})
				continue
			}
			manifest.Symbols = append(manifest.Symbols, Symbol{
				ID: id, File: declaration.File, SignatureSHA256: facts.signature,
				BodySHA256: facts.body, Exported: declaration.Exported,
			})
			known[declarationKey(declaration)] = id
			if boundary, ok := declarationUncertainty(declaration, graph.context.ID, id); ok {
				manifest.Uncertainty = append(manifest.Uncertainty, boundary)
			}
		}
		for _, edge := range graph.references {
			from, fromKnown := known[declarationKey(edge.From)]
			to, toKnown := known[declarationKey(edge.To)]
			if !fromKnown || !toKnown {
				continue
			}
			manifest.References = append(manifest.References, Reference{From: from, To: to, Kind: referenceKind(to.Kind)})
		}
	}
	return codec.New(manifest)
}

type fingerprints struct {
	signature string
	body      string
}

func profileFingerprints(profile codeprofile.Profile) map[string]fingerprints {
	result := make(map[string]fingerprints, len(profile.Functions))
	for _, function := range profile.Functions {
		if function.AdvisoryClass == "test" {
			continue
		}
		_, receiver, signature, body := function.StructuralFingerprints()
		result[function.File+"\x00"+receiver+"\x00"+function.Name] = fingerprints{signature: signature, body: body}
	}
	return result
}

func buildContext(selection repoanalysis.BuildSelection) (BuildContext, error) {
	goos, goarch, found := strings.Cut(selection.Context, "/")
	if !found || strings.TrimSpace(goos) == "" || strings.TrimSpace(goarch) == "" || strings.Contains(goarch, "/") {
		return BuildContext{}, errors.New("code manifest: build selection lacks an exact GOOS/GOARCH context")
	}
	return BuildContext{ID: selection.Context, GOOS: goos, GOARCH: goarch}, nil
}

func manifestFile(source repoanalysis.GoFile, graphs []contextGraph) (File, []Uncertainty, error) {
	generated, err := source.Generated()
	if err != nil {
		return File{}, nil, fmt.Errorf("code manifest: generated status for %s: %w", source.Path, err)
	}
	expression, err := source.BuildExpression()
	if err != nil {
		return File{}, nil, fmt.Errorf("code manifest: build expression for %s: %w", source.Path, err)
	}
	var uncertainty []Uncertainty
	file := File{
		Path: source.Path, ContentID: source.ContentID, Package: path.Dir(source.Path),
		BuildExpression: expression, Generated: generated, Test: source.Test,
	}
	for _, graph := range graphs {
		selected, known := graph.selection.Files[source.Path]
		if selected {
			file.SelectedContexts = append(file.SelectedContexts, graph.context.ID)
		}
		if !known {
			uncertainty = append(uncertainty, Uncertainty{
				Kind: UncertaintyBuildSelection, Path: source.Path, Context: graph.context.ID,
				Reason: "build context has no selection decision for the source file",
			})
		}
	}
	if len(file.SelectedContexts) == 0 {
		uncertainty = append(uncertainty, Uncertainty{
			Kind: UncertaintyBuildSelection, Path: source.Path,
			Reason: "no declared build context selects the source file",
		})
	}
	if generated {
		uncertainty = append(uncertainty, Uncertainty{Kind: UncertaintyGenerated, Path: source.Path, Reason: "generated source requires generator authority"})
	}
	return file, uncertainty, nil
}

func functionKind(kind string) (SymbolKind, bool) {
	switch kind {
	case "function":
		return SymbolFunction, true
	case "method":
		return SymbolMethod, true
	default:
		return "", false
	}
}

func referenceKind(kind SymbolKind) ReferenceKind {
	if kind == SymbolFunction || kind == SymbolMethod {
		return ReferenceCall
	}
	return ReferenceUse
}

func declarationKey(value codeprofile.ConsumerDeclaration) string {
	return value.Package + "\x00" + value.File + "\x00" + value.Kind + "\x00" + value.Receiver + "\x00" + value.Name
}

func declarationUncertainty(declaration codeprofile.ConsumerDeclaration, context string, symbol SymbolID) (Uncertainty, bool) {
	kind := UncertaintyKind("")
	switch declaration.Boundary {
	case "reflection":
		kind = UncertaintyReflection
	case "cgo":
		kind = UncertaintyCgo
	case "method-dispatch", "interface":
		kind = UncertaintyInterface
	case "generated":
		kind = UncertaintyGenerated
	case "build-variant":
		kind = UncertaintyBuildSelection
	case "external", "command":
		kind = UncertaintyExternal
	default:
		return Uncertainty{}, false
	}
	return Uncertainty{
		Kind: kind, Path: declaration.File, Context: context, Symbol: &symbol,
		Reason: "consumer census boundary: " + declaration.Boundary,
	}, true
}
