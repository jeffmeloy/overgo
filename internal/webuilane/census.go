package webuilane

import (
	"regexp"
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
	UnnamedControls int `json:"unnamed_controls"` // el("input"/"select"/"textarea") without aria-label, a file/checkbox/hidden type or a wrapping label on the line (a placeholder is not a name)
	UnnamedButtons  int `json:"unnamed_buttons"`  // el("button", {...}) closed on its attributes without text, aria-label, title or a child
	InlineStyles    int `json:"inline_styles"`    // style: "..." attributes in el() calls
	NestedTernaries int `json:"nested_ternaries"` // a ternary inside a ternary's branch on one line
	TimerLiterals   int `json:"timer_literals"`   // setTimeout/setInterval with a numeric literal
	DebtMarkers     int `json:"debt_markers"`     // TODO/FIXME/HACK/XXX
}

var (
	silentFallbackPattern = regexp.MustCompile(`\.catch\(\([^)]*\) => (?:null|\{\s*\}|\[\]|false|undefined|""|0)\)(\s*/\*)?`)
	windowDialogPattern   = regexp.MustCompile(`(?:^|[\s(=!&|,])(?:window\.)?(?:alert|confirm|prompt)\(`)
	controlPattern        = regexp.MustCompile(`el\("(?:input|select|textarea)"`)
	buttonPattern         = regexp.MustCompile(`el\("button"`)
	namedControlPattern   = regexp.MustCompile(`aria-label|type: "(?:file|checkbox|hidden)"`)
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
