package codemanifest

import (
	"errors"
	"fmt"
	"path"
	"strings"

	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
)

const (
	analyzerName    = "overgo-codeprofile"
	analyzerVersion = "consumer-graph-v1"
)

// Generate composes a canonical manifest from the repository's existing
// source snapshot, function profile, build selection, and consumer graph.
func Generate(snapshot repoanalysis.SourceSnapshot, selection repoanalysis.BuildSelection) (Manifest, error) {
	context, err := buildContext(selection)
	if err != nil {
		return Manifest{}, err
	}
	profile, err := codeprofile.Build(snapshot)
	if err != nil {
		return Manifest{}, fmt.Errorf("code manifest: profile: %w", err)
	}
	declarations, edges, _, err := codeprofile.ProductionConsumerGraph(snapshot, selection)
	if err != nil {
		return Manifest{}, fmt.Errorf("code manifest: consumer graph: %w", err)
	}
	manifest := Manifest{
		Version: Version, SourceIdentity: snapshot.Identity(),
		Analyzer:      Analyzer{Name: analyzerName, Version: analyzerVersion},
		BuildContexts: []BuildContext{context},
	}
	packages := declarationPackages(declarations)
	for _, source := range snapshot.Files {
		file, boundaries, err := manifestFile(source, selection, context.ID, packages[source.Path])
		if err != nil {
			return Manifest{}, err
		}
		manifest.Files = append(manifest.Files, file)
		manifest.Uncertainty = append(manifest.Uncertainty, boundaries...)
	}
	fingerprints := profileFingerprints(profile)
	known := map[string]SymbolID{}
	for _, declaration := range declarations {
		if !selection.Files[declaration.File] {
			continue
		}
		kind, ok := functionKind(declaration.Kind)
		if !ok {
			continue
		}
		key := declaration.File + "\x00" + declaration.Receiver + "\x00" + declaration.Name
		facts, found := fingerprints[key]
		id := SymbolID{Package: declaration.Package, Receiver: declaration.Receiver, Name: declaration.Name, Kind: kind}
		if !found {
			manifest.Uncertainty = append(manifest.Uncertainty, Uncertainty{
				Kind: UncertaintyAnalysis, Path: declaration.File, Symbol: &id,
				Reason: "function declaration has no profiled body and signature",
			})
			continue
		}
		manifest.Symbols = append(manifest.Symbols, Symbol{
			ID: id, File: declaration.File, SignatureSHA256: facts.signature,
			BodySHA256: facts.body, Exported: declaration.Exported,
		})
		known[declarationKey(declaration)] = id
		if boundary, ok := declarationUncertainty(declaration, id); ok {
			manifest.Uncertainty = append(manifest.Uncertainty, boundary)
		}
	}
	for _, edge := range edges {
		from, fromKnown := known[declarationKey(edge.From)]
		to, toKnown := known[declarationKey(edge.To)]
		if !fromKnown || !toKnown {
			continue
		}
		manifest.References = append(manifest.References, Reference{From: from, To: to, Kind: referenceKind(to.Kind)})
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

func declarationPackages(declarations []codeprofile.ConsumerDeclaration) map[string]string {
	packages := map[string]string{}
	for _, declaration := range declarations {
		if packages[declaration.File] == "" {
			packages[declaration.File] = declaration.Package
		}
	}
	return packages
}

func manifestFile(source repoanalysis.GoFile, selection repoanalysis.BuildSelection, contextID, declaredPackage string) (File, []Uncertainty, error) {
	generated, err := source.Generated()
	if err != nil {
		return File{}, nil, fmt.Errorf("code manifest: generated status for %s: %w", source.Path, err)
	}
	expression, err := source.BuildExpression()
	if err != nil {
		return File{}, nil, fmt.Errorf("code manifest: build expression for %s: %w", source.Path, err)
	}
	packagePath := selection.Packages[source.Path]
	if packagePath == "" {
		packagePath = declaredPackage
	}
	var uncertainty []Uncertainty
	if packagePath == "" {
		packagePath = path.Dir(source.Path)
		uncertainty = append(uncertainty, Uncertainty{
			Kind: UncertaintyBuildSelection, Path: source.Path,
			Reason: "build selection and declaration census do not identify the package",
		})
	}
	file := File{
		Path: source.Path, ContentID: source.ContentID, Package: packagePath,
		BuildExpression: expression, Generated: generated, Test: source.Test,
	}
	selected, known := selection.Files[source.Path]
	if selected {
		file.SelectedContexts = []string{contextID}
	}
	if !known || !selected {
		reason := "build context excludes the source file"
		if !known {
			reason = "build context has no selection decision for the source file"
		}
		uncertainty = append(uncertainty, Uncertainty{Kind: UncertaintyBuildSelection, Path: source.Path, Reason: reason})
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

func declarationUncertainty(declaration codeprofile.ConsumerDeclaration, symbol SymbolID) (Uncertainty, bool) {
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
	return Uncertainty{Kind: kind, Path: declaration.File, Symbol: &symbol, Reason: "consumer census boundary: " + declaration.Boundary}, true
}
