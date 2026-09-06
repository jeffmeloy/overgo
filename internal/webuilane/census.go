package webuilane

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/apimanifest"
)

// Census is the client's implementation size at one tree: the facts the
// simplification report compares between the campaign's fork and its head.
// The JavaScript counts restate the composer budget ratchet's definitions
// (internal/server/webui_composer_budget_test.go) so the report and the
// ceiling agree on every number.
type Census struct {
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
	Review                   // the review criteria, summed over every *.js
}

// Review is the client review criteria measured over JavaScript source
// (owner rule 2026-09-06: review concepts are owned as measures with
// ratchets, never as rule prose): silent failures, dialogs that leave the
// page, controls without an accessible name, inline styling, nested
// conditionals, timer literals and debt markers.
type Review struct {
	SilentFallbacks int `json:"silent_fallbacks"` // .catch swallowing to null/{}/[]/false without a reviewed comment after it
	WindowDialogs   int `json:"window_dialogs"`   // alert/confirm/prompt calls
	UnnamedControls int `json:"unnamed_controls"` // el("input"/"select") without aria-label, placeholder, a file/checkbox/hidden type or a wrapping label on the line
	UnnamedButtons  int `json:"unnamed_buttons"`  // el("button", {...}) closed on its attributes without text, aria-label or title
	InlineStyles    int `json:"inline_styles"`    // style: "..." attributes in el() calls
	NestedTernaries int `json:"nested_ternaries"` // a ternary inside a ternary's branch on one line
	TimerLiterals   int `json:"timer_literals"`   // setTimeout/setInterval with a numeric literal
	DebtMarkers     int `json:"debt_markers"`     // TODO/FIXME/HACK/XXX
}

var (
	silentFallbackPattern = regexp.MustCompile(`\.catch\(\([^)]*\) => (?:null|\{\s*\}|\[\]|false|undefined|""|0)\)(\s*/\*)?`)
	windowDialogPattern   = regexp.MustCompile(`(?:^|[\s(=!&|,])(?:window\.)?(?:alert|confirm|prompt)\(`)
	controlPattern        = regexp.MustCompile(`el\("(?:input|select)"[^\n]*`)
	buttonPattern         = regexp.MustCompile(`el\("button", \{[^}]*\}\)`)
	inlineStylePattern    = regexp.MustCompile(`style: "`)
	nestedTernaryPattern  = regexp.MustCompile(`\?[^:\n]*\?[^:\n]*:`)
	timerLiteralPattern   = regexp.MustCompile(`set(?:Timeout|Interval)\([^\n]*?,\s*\d+\)`)
	debtMarkerPattern     = regexp.MustCompile(`\b(?:TODO|FIXME|HACK|XXX)\b`)
)

// ReviewMeasures measures one JavaScript source against the review criteria.
func ReviewMeasures(source string) Review {
	var review Review
	for _, match := range silentFallbackPattern.FindAllStringSubmatch(source, everyMatch) {
		if match[1] == "" {
			review.SilentFallbacks++
		}
	}
	review.WindowDialogs = len(windowDialogPattern.FindAllString(source, everyMatch))
	for line := range strings.SplitSeq(source, "\n") {
		for _, at := range controlPattern.FindAllStringIndex(line, everyMatch) {
			call, before := line[at[0]:at[1]], line[:at[0]]
			if !strings.Contains(call, "aria-label") && !strings.Contains(call, "placeholder") && !strings.Contains(call, `type: "file"`) &&
				!strings.Contains(call, `type: "checkbox"`) && !strings.Contains(call, `type: "hidden"`) && !strings.Contains(before, `el("label"`) {
				review.UnnamedControls++
			}
		}
	}
	for _, call := range buttonPattern.FindAllString(source, everyMatch) {
		if !strings.Contains(call, "text:") && !strings.Contains(call, "aria-label") && !strings.Contains(call, "title:") {
			review.UnnamedButtons++
		}
	}
	review.InlineStyles = len(inlineStylePattern.FindAllString(source, everyMatch))
	review.NestedTernaries = len(nestedTernaryPattern.FindAllString(source, everyMatch))
	review.TimerLiterals = len(timerLiteralPattern.FindAllString(source, everyMatch))
	review.DebtMarkers = len(debtMarkerPattern.FindAllString(source, everyMatch))
	return review
}

// add sums another source's measures into this review.
func (review *Review) add(other Review) {
	review.SilentFallbacks += other.SilentFallbacks
	review.WindowDialogs += other.WindowDialogs
	review.UnnamedControls += other.UnnamedControls
	review.UnnamedButtons += other.UnnamedButtons
	review.InlineStyles += other.InlineStyles
	review.NestedTernaries += other.NestedTernaries
	review.TimerLiterals += other.TimerLiterals
	review.DebtMarkers += other.DebtMarkers
}

const (
	webuiDirectory = "internal/server/webui"
	manifestPath   = "docs/api_manifest.json"
	// everyMatch asks the regexp package for all matches (a negative count).
	everyMatch = -1
)

var (
	fetchSitePattern     = regexp.MustCompile(`\bfetch\(`)
	clientPathPattern    = regexp.MustCompile(`"(/[A-Za-z0-9_./-]+)`)
	scriptTagPattern     = regexp.MustCompile(`<script[^>]*\bsrc=`)
	scriptLiteralPattern = regexp.MustCompile(`"/[A-Za-z0-9_./-]+\.js"`)
)

// MeasureTree measures the client under root: a repository checkout or an
// extracted slice of one holding internal/server/webui and the API
// manifest. label names the tree in the report (its commit).
func MeasureTree(root, label string) (Census, error) {
	census := Census{Tree: label}
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
			census.Review.add(ReviewMeasures(source))
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
		return Census{}, fmt.Errorf("measure %s: %w", webui, err)
	}
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(manifestPath)))
	if err != nil {
		return Census{}, err
	}
	manifest, err := apimanifest.Parse(data)
	if err != nil {
		return Census{}, fmt.Errorf("measure %s: %w", root, err)
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

// BrowserTestPrefix names the browser acceptance tests: the lane runs them
// by this prefix, and a plan verify naming one without the lane is refused.
const BrowserTestPrefix = "TestWebUIBrowser"

// LaneVerdict judges one lane run from its output: a test the run skipped
// or a run in which no test passed is no evidence, and every required
// line (a journey leg's log) must have been written; nil is the pass.
func LaneVerdict(output string, required []string) error {
	proven, _ := LaneObservations(output)
	passed := 0
	for line := range strings.SplitSeq(output, "\n") {
		trimmed := strings.TrimSpace(line)
		if skipped, ok := strings.CutPrefix(trimmed, "--- SKIP: "); ok {
			return fmt.Errorf("webui lane: %s was skipped, so it proves nothing", skipped)
		}
		if strings.HasPrefix(trimmed, "--- PASS: ") {
			passed++
		}
	}
	if passed == 0 {
		return errors.New("webui lane: no browser test passed")
	}
	for _, text := range required {
		if !slices.ContainsFunc(proven, func(line string) bool { return strings.Contains(line, text) }) {
			return fmt.Errorf("webui lane: the run did not write the required evidence %q", text)
		}
	}
	return nil
}

// LaneObservations reads the lane's own output: the tests that passed and
// the journey lines naming a leg are the behaviours proven; the lines the
// lane marked UNAVAILABLE are the behaviours it could not observe.
func LaneObservations(output string) (proven, unobserved []string) {
	for line := range strings.SplitSeq(output, "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "--- PASS: "):
			proven = append(proven, strings.TrimPrefix(trimmed, "--- PASS: "))
		case strings.HasPrefix(trimmed, "webui lane: UNAVAILABLE "):
			unobserved = append(unobserved, strings.TrimPrefix(trimmed, "webui lane: UNAVAILABLE "))
		case strings.Contains(trimmed, "journey:") || strings.Contains(trimmed, " leg"):
			if _, after, ok := strings.Cut(trimmed, ": "); ok {
				proven = append(proven, after)
			}
		}
	}
	return proven, unobserved
}

// SimplificationReport renders the campaign's simplification report: every
// census measure at the fork beside the head with its change, then the
// behaviours the acceptance lane proved on the head and the ones it could
// not observe. The text is a function of its inputs alone.
func SimplificationReport(fork, head Census, proven, unobserved []string) string {
	measures := []struct {
		name string
		get  func(Census) int
	}{
		{"Shells (HTML documents)", func(c Census) int { return c.Shells }},
		{"Hand-listed scripts", func(c Census) int { return c.ListedScripts }},
		{"Tab modules (mod/*.js)", func(c Census) int { return c.Modules }},
		{"JavaScript lines", func(c Census) int { return c.JavaScriptLines }},
		{"fetch call sites", func(c Census) int { return c.FetchSites }},
		{"Stream reader sites", func(c Census) int { return c.StreamReaderSites }},
		{"api.stream sites", func(c Census) int { return c.APIStreamSites }},
		{"API manifest routes", func(c Census) int { return c.Routes }},
		{"Bearer-authenticated routes", func(c Census) int { return c.BearerRoutes }},
		{"Routes the client names", func(c Census) int { return c.ClientRoutes }},
		{"Largest file (lines)", func(c Census) int { return c.LargestFileLines }},
		{"Silent fallbacks", func(c Census) int { return c.SilentFallbacks }},
		{"Window dialogs", func(c Census) int { return c.WindowDialogs }},
		{"Controls without a name", func(c Census) int { return c.UnnamedControls }},
		{"Buttons without a name", func(c Census) int { return c.UnnamedButtons }},
		{"Inline style attributes", func(c Census) int { return c.InlineStyles }},
		{"Nested ternaries", func(c Census) int { return c.NestedTernaries }},
		{"Timer literals", func(c Census) int { return c.TimerLiterals }},
		{"Debt markers", func(c Census) int { return c.DebtMarkers }},
	}
	var report strings.Builder
	fmt.Fprintf(&report, "# Simplification report: %s to %s\n\n", fork.Tree, head.Tree)
	report.WriteString("Generated by `go run ./cmd/webui-lane -report`; the counts restate the\n")
	report.WriteString("composer budget ratchet's definitions and the API manifest.\n\n")
	fmt.Fprintf(&report, "| Measure | Fork %s | Head %s | Change |\n| --- | ---: | ---: | ---: |\n", fork.Tree, head.Tree)
	for _, measure := range measures {
		before, after := measure.get(fork), measure.get(head)
		fmt.Fprintf(&report, "| %s | %d | %d | %+d |\n", measure.name, before, after, after-before)
	}
	report.WriteString("\n## Behaviours the acceptance lane proved on the head\n\n")
	for _, behaviour := range proven {
		fmt.Fprintf(&report, "- %s\n", behaviour)
	}
	if len(unobserved) > 0 {
		report.WriteString("\n## Behaviours the lane could not observe\n\n")
		for _, behaviour := range unobserved {
			fmt.Fprintf(&report, "- %s\n", behaviour)
		}
	}
	return report.String()
}
