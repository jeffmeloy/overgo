package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"overgo/internal/apimanifest"
	"overgo/internal/jsonfile"
	"overgo/internal/webuilane"
)

const (
	webuiDirectory    = "internal/server/webui"
	webuiManifestPath = "docs/api_manifest.json"
	// regexp uses a negative count to request every match.
	everyMatch = -1
)

// webuiCensus measures source size and review debt, not behavioral acceptance.
type webuiCensus struct {
	Tree              string `json:"tree"`
	Shells            int    `json:"shells"`              // HTML documents under webui/
	ListedScripts     int    `json:"listed_scripts"`      // scripts listed by hand: shell script tags with a source, loader path literals
	Modules           int    `json:"modules"`             // webui/mod/*.js self-registering tabs
	JavaScriptLines   int    `json:"javascript_lines"`    // newline count over every *.js under webui/
	FetchSites        int    `json:"fetch_sites"`         // fetch( call sites
	StreamReaderSites int    `json:"stream_reader_sites"` // .getReader() sites
	APIStreamSites    int    `json:"api_stream_sites"`    // api.stream( sites
	Routes            int    `json:"routes"`              // distinct route paths in the API manifest
	BearerRoutes      int    `json:"bearer_routes"`       // distinct bearer-authenticated route paths
	ClientRoutes      int    `json:"client_routes"`       // manifest routes the client names by literal path
	LargestFileLines  int    `json:"largest_file_lines"`  // newline count of the largest *.js under webui/
	webuilane.Review         // the review criteria, summed over every *.js
}

var (
	fetchSitePattern     = regexp.MustCompile(`\bfetch\(`)
	clientPathPattern    = regexp.MustCompile(`"(/[A-Za-z0-9_./-]+)`)
	scriptTagPattern     = regexp.MustCompile(`<script[^>]*\bsrc=`)
	scriptLiteralPattern = regexp.MustCompile(`"/[A-Za-z0-9_./-]+\.js"`)
)

// measureWebUITree measures the client under root: a repository checkout or an
// extracted slice of one holding internal/server/webui and the API
// manifest. label names the tree in the report (its commit).
func measureWebUITree(root, label string) (webuiCensus, error) {
	census := webuiCensus{Tree: label}
	webui := filepath.Join(root, filepath.FromSlash(webuiDirectory))
	named := map[string]bool{}
	err := fs.WalkDir(os.DirFS(webui), ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		data, err := os.ReadFile(filepath.Join(webui, filepath.FromSlash(path)))
		if err != nil {
			return err
		}
		source := string(data)
		switch filepath.Ext(path) {
		case ".html":
			census.Shells++
			census.ListedScripts += len(scriptTagPattern.FindAllString(source, everyMatch))
		case ".js":
			if strings.HasPrefix(path, "mod/") {
				census.Modules++
			}
			census.ListedScripts += len(scriptLiteralPattern.FindAllString(source, everyMatch))
			census.JavaScriptLines += strings.Count(source, "\n")
			census.LargestFileLines = max(census.LargestFileLines, strings.Count(source, "\n"))
			census.Review.Add(webuilane.ReviewMeasures(source))
			census.FetchSites += len(fetchSitePattern.FindAllString(source, everyMatch))
			census.StreamReaderSites += strings.Count(source, ".getReader()")
			census.APIStreamSites += strings.Count(source, "api.stream(")
			for _, match := range clientPathPattern.FindAllStringSubmatch(source, everyMatch) {
				named[match[1]] = true
			}
		}
		return nil
	})
	if err != nil {
		return webuiCensus{}, fmt.Errorf("measure %s: %w", webui, err)
	}
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(webuiManifestPath)))
	if err != nil {
		return webuiCensus{}, err
	}
	manifest, err := apimanifest.Parse(data)
	if err != nil {
		return webuiCensus{}, fmt.Errorf("measure %s: %w", root, err)
	}
	distinct := map[string]bool{}
	bearer := map[string]bool{}
	for _, route := range manifest.Routes {
		distinct[route.Path] = true
		if route.Authentication == "bearer" {
			bearer[route.Path] = true
		}
	}
	for path := range distinct {
		if named[path] {
			census.ClientRoutes++
		}
	}
	census.Routes, census.BearerRoutes = len(distinct), len(bearer)
	return census, nil
}

func writeWebUIReport(specPath string, output io.Writer) error {
	type tree struct {
		Root  string `json:"root"`
		Label string `json:"label"`
	}
	var spec struct {
		Before tree `json:"before"`
		After  tree `json:"after"`
	}
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	if strings.TrimSpace(spec.Before.Root) == "" || strings.TrimSpace(spec.After.Root) == "" {
		return errors.New("webui report: before.root and after.root are required")
	}
	before, err := measureWebUITree(spec.Before.Root, spec.Before.Label)
	if err != nil {
		return err
	}
	after, err := measureWebUITree(spec.After.Root, spec.After.Label)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(struct {
		Before webuiCensus `json:"before"`
		After  webuiCensus `json:"after"`
	}{before, after})
}
