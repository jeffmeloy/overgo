package webuilane

import (
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// Review is the client review criteria measured over JavaScript source
// (owner rule 2026-09-06: review concepts are owned as measures with
// ratchets, never as rule prose): silent failures, dialogs that leave the
// page, controls without an accessible name, inline styling, nested
// conditionals, timer literals and debt markers.
type Review struct {
	SilentFallbacks int `json:"silent_fallbacks"` // .catch swallowing to null/{}/[]/false without a reviewed comment after it
	WindowDialogs   int `json:"window_dialogs"`   // alert/confirm/prompt calls
	UnnamedControls int `json:"unnamed_controls"` // el("input"/"select") without aria-label, placeholder, a file/checkbox/hidden type or a wrapping label on the line
	UnnamedButtons  int `json:"unnamed_buttons"`  // el("button", {...}) closed on its attributes without text, aria-label, title or a child
	InlineStyles    int `json:"inline_styles"`    // style: "..." attributes in el() calls
	NestedTernaries int `json:"nested_ternaries"` // a ternary inside a ternary's branch on one line
	TimerLiterals   int `json:"timer_literals"`   // setTimeout/setInterval with a numeric literal
	DebtMarkers     int `json:"debt_markers"`     // TODO/FIXME/HACK/XXX
}

var (
	silentFallbackPattern = regexp.MustCompile(`\.catch\(\([^)]*\) => (?:null|\{\s*\}|\[\]|false|undefined|""|0)\)(\s*/\*)?`)
	windowDialogPattern   = regexp.MustCompile(`(?:^|[\s(=!&|,])(?:window\.)?(?:alert|confirm|prompt)\(`)
	controlPattern        = regexp.MustCompile(`el\("(?:input|select)"`)
	buttonPattern         = regexp.MustCompile(`el\("button"`)
	namedControlPattern   = regexp.MustCompile(`aria-label|placeholder|type: "(?:file|checkbox|hidden)"`)
	namedButtonPattern    = regexp.MustCompile(`aria-label|title:|\btext[:,}]|\}, \S`)
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
	for _, at := range controlPattern.FindAllStringIndex(source, everyMatch) {
		// before: the call's own line up to the call, where a wrapping label would be written.
		call, before := elCall(source, at[0]), source[:at[0]]
		if lineStart := strings.LastIndexByte(before, '\n'); lineStart >= 0 {
			before = before[lineStart:]
		}
		if !namedControlPattern.MatchString(call) && !strings.Contains(before, `el("label"`) {
			review.UnnamedControls++
		}
	}
	for _, at := range buttonPattern.FindAllStringIndex(source, everyMatch) {
		if !namedButtonPattern.MatchString(elCall(source, at[0])) {
			review.UnnamedButtons++
		}
	}
	review.InlineStyles = len(inlineStylePattern.FindAllString(source, everyMatch))
	review.NestedTernaries = len(nestedTernaryPattern.FindAllString(source, everyMatch))
	review.TimerLiterals = len(timerLiteralPattern.FindAllString(source, everyMatch))
	review.DebtMarkers = len(debtMarkerPattern.FindAllString(source, everyMatch))
	return review
}

// elCall returns the el(...) call text starting at start, to its balanced closing
// parenthesis (the whole call, however many lines it spans).
func elCall(source string, start int) string {
	depth := 0
	for index := start; index < len(source); index++ {
		switch source[index] {
		case '(':
			depth++
		case ')':
			if depth--; depth == 0 {
				return source[start : index+1]
			}
		}
	}
	return source[start:]
}

// Add sums another source's measures into this review.
func (review *Review) Add(other Review) {
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
	// everyMatch asks the regexp package for all matches (a negative count).
	everyMatch = -1
)

// BrowserTestPrefix names the browser acceptance tests: the lane runs them
// by this prefix, and a plan verify naming one without the lane is refused.
const BrowserTestPrefix = "TestWebUIBrowser"

// ModelJourneyPrefix names the browser journeys that build, serve or hash a
// model: the lane runs them only in its -journeys mode, so the page
// acceptances run against retained receipts alone.
const ModelJourneyPrefix = "TestModelJourney"

// ModelJourneyEnvironment is set by the lane's -journeys mode; a journey
// skips without it, since outside the lane it proves nothing.
const ModelJourneyEnvironment = "OVERGO_MODEL_JOURNEY"

// LaneVerdict judges one lane run from its output: a test the run skipped
// or a run in which no test passed is no evidence, and every required
// line (a journey leg's log) must have been written; nil is the pass.
func LaneVerdict(output string, required []string) error {
	var proven []string
	passed := 0
	for line := range strings.SplitSeq(output, "\n") {
		trimmed := strings.TrimSpace(line)
		if skipped, ok := strings.CutPrefix(trimmed, "--- SKIP: "); ok {
			return fmt.Errorf("webui lane: %s was skipped, so it proves nothing", skipped)
		}
		if name, ok := strings.CutPrefix(trimmed, "--- PASS: "); ok {
			passed++
			proven = append(proven, name)
		} else if !strings.HasPrefix(trimmed, "webui lane: UNAVAILABLE ") && (strings.Contains(trimmed, "journey:") || strings.Contains(trimmed, " leg")) {
			if _, after, ok := strings.Cut(trimmed, ": "); ok {
				proven = append(proven, after)
			}
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

// FailureLines names each failed test with the last line its own source
// wrote before its verdict and that line's indented continuation: the
// failing step and its page state, which a bounded tail of the whole run
// would lose behind the tests after it.
func FailureLines(output string) []string {
	var failures []string
	var last string
	for line := range strings.SplitSeq(output, "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "=== RUN "):
			last = ""
		case strings.Contains(trimmed, "_test.go:"):
			last = trimmed
		case strings.HasPrefix(trimmed, "--- FAIL: "):
			name, _, _ := strings.Cut(strings.TrimPrefix(trimmed, "--- FAIL: "), " (")
			failures = append(failures, name+": "+last)
		case last != "" && strings.HasPrefix(line, "        ") && trimmed != "":
			// The step's name leads the line; a page dump behind it is cut so
			// the name survives a caller's bounded tail.
			if len(last) < failureLineBytes {
				last += " " + trimmed
				if len(last) > failureLineBytes {
					last = strings.ToValidUTF8(last[:failureLineBytes], "") + "…"
				}
			}
		}
	}
	return failures
}

// failureLineBytes bounds one reported failure so several fit a caller's
// diagnostic tail beside one another.
const failureLineBytes = 800
