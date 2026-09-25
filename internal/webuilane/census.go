package webuilane

import (
	"slices"
	"strings"
	"unicode"

	"overgo/internal/extent"
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

// ReviewMeasures measures one JavaScript source against the review
// criteria, reading its tokens: a word in a string or a comment is never
// taken for code, and a call is read to its balanced close however many
// lines it spans.
func ReviewMeasures(source string) Review {
	tokens := tokenizeJS(source)
	// code is the source without its comments; commented[i] reports a
	// comment written straight after code[i].
	code := make([]jsToken, 0, len(tokens))
	var commented []bool
	var review Review
	for _, token := range tokens {
		if token.kind != jsComment {
			code = append(code, token)
			commented = append(commented, false)
			continue
		}
		if len(commented) > 0 {
			commented[len(commented)-1] = true
		}
		for word := range strings.FieldsFuncSeq(token.text, func(character rune) bool {
			return character > unicode.MaxASCII || !isIdentifierByte(byte(character))
		}) {
			if slices.Contains(debtMarkers, word) {
				review.DebtMarkers++
			}
		}
	}
	for index, token := range code {
		switch {
		case token.is(jsPunctuator, ".") && index+1 < len(code) && code[index+1].is(jsIdentifier, "catch"):
			if end, silent := silentFallback(code, index+1); silent && !commented[end] {
				review.SilentFallbacks++
			}
		case token.kind == jsIdentifier && slices.Contains(windowDialogs, token.text) && next(code, index, "("):
			if index == 0 || !code[index-1].is(jsPunctuator, ".") || index > 1 && code[index-2].is(jsIdentifier, "window") {
				review.WindowDialogs++
			}
		case token.is(jsIdentifier, "el") && next(code, index, "(") && index+2 < len(code) && code[index+2].kind == jsString:
			call := code[index+1 : closing(code, index+1)+1]
			switch element := code[index+2].value(); {
			case slices.Contains(formControls, element):
				if !controlNamed(call) && !insideLabel(code, index) {
					review.UnnamedControls++
				}
			case element == "button":
				if !buttonNamed(call) {
					review.UnnamedButtons++
				}
			}
		case token.is(jsIdentifier, "style") && index+2 < len(code) && code[index+1].is(jsPunctuator, ":") && code[index+2].kind == jsString:
			review.InlineStyles++
		case (token.is(jsIdentifier, "setTimeout") || token.is(jsIdentifier, "setInterval")) && next(code, index, "("):
			call := arguments(code[index+2 : closing(code, index+1)])
			if len(call) >= extent.PairedExtent {
				if last := call[len(call)-1]; len(last) == extent.SingletonExtent && last[0].kind == jsNumber {
					review.TimerLiterals++
				}
			}
		}
	}
	review.NestedTernaries = nestedTernaries(code)
	return review
}

var (
	debtMarkers   = []string{"TODO", "FIXME", "HACK", "XXX"}
	windowDialogs = []string{"alert", "confirm", "prompt"}
	formControls  = []string{"input", "select", "textarea"}
	// unnamedTypes are the input types that need no accessible name.
	unnamedTypes = []string{"file", "checkbox", "hidden"}
	// swallowedValues are the values a silent catch settles to.
	swallowedValues = []string{"null", "false", "undefined", "0", `""`, "''"}
)

func next(tokens []jsToken, index int, text string) bool {
	return index+1 < len(tokens) && tokens[index+1].is(jsPunctuator, text)
}

// jsCursor walks a token sequence one expected token at a time.
type jsCursor struct {
	tokens []jsToken
	index  int
}

// step moves onto the next token when it is this punctuator.
func (cursor *jsCursor) step(text string) bool {
	if !next(cursor.tokens, cursor.index, text) {
		return false
	}
	cursor.index++
	return true
}

// silentFallback reads .catch((...) => value) settling to an empty value,
// from the catch token, answering the index of the call's closing
// parenthesis.
func silentFallback(code []jsToken, catch int) (end int, silent bool) {
	cursor := jsCursor{tokens: code, index: catch}
	if !cursor.step("(") || !cursor.step("(") {
		return end, silent
	}
	cursor.index = closing(code, cursor.index)
	if !cursor.step("=>") || cursor.index >= len(code)-extent.SingletonExtent {
		return end, silent
	}
	cursor.index++
	switch value := code[cursor.index]; {
	case value.is(jsPunctuator, "{"):
		if !cursor.step("}") {
			return end, silent
		}
	case value.is(jsPunctuator, "["):
		if !cursor.step("]") {
			return end, silent
		}
	case value.kind == jsPunctuator || !slices.Contains(swallowedValues, value.text):
		return end, silent
	}
	if !cursor.step(")") {
		return end, silent
	}
	return cursor.index, true
}

// controlNamed: an accessible name in the call's attributes, or a type
// that needs none.
func controlNamed(call []jsToken) bool {
	for index, token := range call {
		if token.kind == jsString && token.value() == "aria-label" {
			return true
		}
		if token.is(jsIdentifier, "type") && index+2 < len(call) && call[index+1].is(jsPunctuator, ":") &&
			slices.Contains(unnamedTypes, call[index+2].value()) {
			return true
		}
	}
	return false
}

// insideLabel: an el("label") call opened earlier on the control's line.
func insideLabel(code []jsToken, control int) bool {
	for index := control - 1; index >= extent.PairedExtent && code[index].line == code[control].line; index-- {
		if code[index].kind == jsString && code[index].value() == "label" && code[index-1].is(jsPunctuator, "(") && code[index-2].is(jsIdentifier, "el") {
			return true
		}
	}
	return false
}

// buttonNamed: a child after the element and its attributes, or an
// aria-label, title or text attribute.
func buttonNamed(call []jsToken) bool {
	if len(arguments(call[1:len(call)-1])) > extent.PairedExtent {
		return true
	}
	for index, token := range call {
		if token.kind == jsString && token.value() == "aria-label" {
			return true
		}
		if (token.is(jsIdentifier, "title") || token.is(jsIdentifier, "text")) && index+1 < len(call) &&
			slices.Contains([]string{":", ",", "}"}, call[index+1].text) {
			return true
		}
	}
	return false
}

// nestedTernaries counts, line by line, a ternary opened inside another's
// branch before either's colon.
func nestedTernaries(code []jsToken) (count int) {
	var open, nested bool
	var line int
	for _, token := range code {
		if token.line != line {
			open, nested, line = false, false, token.line
		}
		switch {
		case token.is(jsPunctuator, "?"):
			nested, open = open, true
		case token.is(jsPunctuator, ":") && open:
			if nested {
				count++
			}
			open, nested = false, false
		}
	}
	return count
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
