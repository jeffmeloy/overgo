package latentimage

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"
)

// The MOSS Ultra workflow's prompt mode 2 wraps its enhancer's three generations in ComfyUI string nodes
// (fixtures/comfy/moss-ultra/requests/text-to-image.json, subgraph 707:703). Each node's rule is Go string code below,
// held to ComfyUI's nodes on the same text (moss-ultra-sections/prompt-strings); a node's number names it in the
// request. The rules read text as Python's re reads it: \s is Unicode whitespace where a pattern sets no ASCII flag
// (pythonSpace), and under (?a) only ASCII whitespace, ASCII word characters and ASCII case. They wait beside their test
// until moss-ultra-sections/prompt-modes runs them about the enhancer's chat, this file then named without _test.

// mossEnhancer is the workflow's enhancer data, verbatim from its request (scripts/python_golden_references/
// comfy_prompt_strings_golden.py): the three system prompts, the three request templates, and the negative's backup.
//
//go:embed moss_enhancer.json
var mossEnhancerJSON []byte

// enhancerData is the enhancer's prompts and templates.
type enhancerData struct {
	PositiveSystem          string `json:"positive_system"`
	NegativeSystemPhoto     string `json:"negative_system_photo"`
	NegativeSystemProtected string `json:"negative_system_protected"`
	PositiveRequest         string `json:"positive_request"`
	RetryRequest            string `json:"retry_request"`
	NegativeRequest         string `json:"negative_request"`
	NegativeBackup          string `json:"negative_backup"`
}

// mossEnhancer reads the embedded enhancer data.
func mossEnhancer() (enhancerData, error) {
	var data enhancerData
	if err := json.Unmarshal(mossEnhancerJSON, &data); err != nil {
		return enhancerData{}, fmt.Errorf("latent image: the MOSS enhancer's data: %w", err)
	}
	return data, nil
}

// enhancerWords is how many words the first positive answer must hold before the enhancer keeps it (451's {449} after
// one word), and enhancerPhrases how many phrases of the negative it keeps (472's {0,19} after one).
const (
	enhancerWords   = 450
	enhancerPhrases = 20
)

// enhancerFormat is a request template filled as Python's str.format fills it with the fields a, b and c, the only
// ones the workflow's templates name.
func enhancerFormat(template, a, b, c string) string {
	return strings.NewReplacer("{a}", a, "{b}", b, "{c}", c).Replace(template)
}

// hasEnhancerWords is 451: whether text holds at least enhancerWords words, each a run of anything but whitespace.
func hasEnhancerWords(text string) bool {
	return len(strings.FieldsFunc(text, pythonSpace)) >= enhancerWords
}

// noNegativePhrases are what 469 reads as no negative at all, beside nothing.
var noNegativePhrases = []string{"none", "n/a", "na", "null", "no negative prompt", "no negative prompts", "no negative prompt needed",
	"no negative prompts needed", "no exclusions", "no exclusions needed"}

// isNoNegative is 469: whether text is, whitespace aside, nothing or one of noNegativePhrases in any case, with at
// most one full stop or exclamation mark after it.
func isNoNegative(text string) bool {
	text = strings.TrimFunc(text, pythonSpace)
	if trimmed, cut := strings.CutSuffix(text, "."); cut {
		text = trimmed
	} else if trimmed, cut := strings.CutSuffix(text, "!"); cut {
		text = trimmed
	}
	text = strings.TrimRightFunc(text, pythonSpace)
	if text == "" {
		return true
	}
	for _, phrase := range noNegativePhrases {
		if strings.EqualFold(text, phrase) {
			return true
		}
	}
	return false
}

// firstPhrases is 472: the text's first phrases, at most limit, each a run between commas starting and ending in
// neither a comma nor whitespace, from the first through the last as the text holds them; ending at a phrase that is
// empty, and empty where the first is.
func firstPhrases(text string, limit int) string {
	start := len(text) - len(strings.TrimLeftFunc(text, pythonSpace))
	end, at := start, start
	for count := range limit {
		if count > 0 {
			// The comma before the next phrase, whitespace about it.
			next := strings.TrimLeftFunc(text[at:], pythonSpace)
			if !strings.HasPrefix(next, ",") {
				break
			}
			at = len(text) - len(next) + 1
			at = len(text) - len(strings.TrimLeftFunc(text[at:], pythonSpace))
		}
		phrase := text[at:]
		if comma := strings.IndexByte(phrase, ','); comma >= 0 {
			phrase = phrase[:comma]
		}
		phrase = strings.TrimRightFunc(phrase, pythonSpace)
		if phrase == "" {
			break
		}
		at += len(phrase)
		end = at
	}
	return text[start:end]
}

// withoutDuplicates is 473: each comma-separated item, whitespace about its phrase, removed with its comma where the
// same phrase stands alone as a later item, so the last of equal items stays.
func withoutDuplicates(text string) string {
	items := strings.Split(text, ",")
	var kept strings.Builder
	for index, item := range items {
		phrase := strings.TrimFunc(item, pythonSpace)
		last := index == len(items)-1
		if !last && phrase != "" {
			later := false
			for _, after := range items[index+1:] {
				if strings.TrimFunc(after, pythonSpace) == phrase {
					later = true
					break
				}
			}
			if later {
				continue
			}
		}
		kept.WriteString(item)
		if !last {
			kept.WriteByte(',')
		}
	}
	return kept.String()
}

// trimSeparators is 480: whitespace and commas taken off both ends.
func trimSeparators(text string) string {
	return strings.TrimFunc(text, func(r rune) bool { return r == ',' || pythonSpace(r) })
}

// The photo test and the guards read text under (?ai): ASCII case, ASCII word characters and ASCII whitespace. A term is
// a matcher: given text and a position, the ends it may reach, every one, so a term followed by more can be tried at
// each.
type term func(text string, at int) []int

func asciiWord(b byte) bool {
	return b == '_' || b >= '0' && b <= '9' || b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z'
}
func asciiSpace(b byte) bool { return b == ' ' || b >= '\t' && b <= '\r' }
func asciiAlnum(b byte) bool { return asciiWord(b) && b != '_' }

// word is literal text in any ASCII case.
func word(literal string) term {
	return func(text string, at int) []int {
		if at+len(literal) <= len(text) && strings.EqualFold(text[at:at+len(literal)], literal) && isASCII(text[at:at+len(literal)]) {
			return []int{at + len(literal)}
		}
		return nil
	}
}

func isASCII(s string) bool {
	for index := range len(s) {
		if s[index] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// words is any of the literals.
func words(literals ...string) term {
	terms := make([]term, len(literals))
	for index, literal := range literals {
		terms[index] = word(literal)
	}
	return either(terms...)
}

// either is any of the terms.
func either(terms ...term) term {
	return func(text string, at int) []int {
		var ends []int
		for _, t := range terms {
			ends = append(ends, t(text, at)...)
		}
		return ends
	}
}

// then is the terms one after another.
func then(terms ...term) term {
	return func(text string, at int) []int {
		ends := []int{at}
		for _, t := range terms {
			var next []int
			for _, end := range ends {
				next = append(next, t(text, end)...)
			}
			if ends = next; len(ends) == 0 {
				return nil
			}
		}
		return ends
	}
}

// maybe is the term or nothing.
func maybe(t term) term { return times(t, 0, 1) }

// times is the term repeated from least to most times.
func times(t term, least, most int) term {
	return func(text string, at int) []int {
		var ends []int
		frontier := []int{at}
		for count := 0; count <= most && len(frontier) > 0; count++ {
			if count >= least {
				ends = append(ends, frontier...)
			}
			var next []int
			for _, end := range frontier {
				next = append(next, t(text, end)...)
			}
			frontier = next
		}
		return ends
	}
}

// run is at least least bytes, each one keep holds.
func run(keep func(byte) bool, least int) term {
	return func(text string, at int) []int {
		var ends []int
		for end := at; ; end++ {
			if end-at >= least {
				ends = append(ends, end)
			}
			if end >= len(text) || !keep(text[end]) {
				return ends
			}
		}
	}
}

// joiner is one of space, underscore and hyphen, as [ _-] reads.
func joiner(b byte) bool { return b == ' ' || b == '_' || b == '-' }

var (
	oneJoiner = run(joiner, 1)
	spaces    = run(asciiSpace, 0)
)

// joined is the parts with [ _-] between them: one or more joiners, or where optional says so at most one ([ _-]?).
func joined(optional bool, parts ...term) term {
	between := oneJoiner
	if optional {
		between = func(text string, at int) []int {
			ends := []int{at}
			if at < len(text) && joiner(text[at]) {
				ends = append(ends, at+1)
			}
			return ends
		}
	}
	sequence := []term{parts[0]}
	for _, part := range parts[1:] {
		sequence = append(sequence, between, part)
	}
	return then(sequence...)
}

var (
	// photoReal is photo[ _-]?real(?:istic|ism), and hyperReal hyper[ _-]?real(?:istic|ism).
	photoReal = joined(true, word("photo"), then(word("real"), words("istic", "ism")))
	hyperReal = joined(true, word("hyper"), then(word("real"), words("istic", "ism")))
	// photoNamed is 474's photographic words: photos?, photographs?, photography, photographic, photoReal, selfies?,
	// snapshots?.
	photoNamed = either(words("photos", "photo", "photographs", "photograph", "photography", "photographic"), photoReal,
		words("selfies", "selfie", "snapshots", "snapshot"))
	// realisticPortrait is 474's (?:realistic|hyper[ _-]?realistic)\s+(?:(?:studio|editorial)\s+)?portraits?.
	realisticPortrait = then(either(word("realistic"), joined(true, word("hyper"), word("realistic"))), run(asciiSpace, 1),
		maybe(then(words("studio", "editorial"), run(asciiSpace, 1))), words("portraits", "portrait"))
	// otherMedium is 474's media that make a picture no photograph, each between no letters or digits.
	otherMedium = either(words("anime", "manga", "manhwa", "cartoons", "cartoon", "cartoonish", "illustrations", "illustration", "illustrated",
		"illustrator", "paintings", "painting", "painted", "painterly", "gouache", "drawings", "drawing", "drawn", "sketches", "sketching", "sketch",
		"comics", "comic"),
		joined(false, words("digital", "official", "concept", "fan", "pixel", "vector"), word("art")),
		joined(true, word("cel"), then(word("shad"), words("ed", "ing"))),
		joined(true, words("2", "3"), word("d")),
		words("cgi", "renders", "rendered", "rendering", "render", "animation", "animated", "watercolour", "watercolor", "sculptures", "sculpture",
			"statues", "statue", "figurines", "figurine"),
		joined(false, word("action"), words("figures", "figure")),
		words("toys", "toy", "posters", "poster", "screens", "screen"),
		joined(false, word("mixed"), word("media")),
		words("hybrid", "collage"))
	// styleReference is 474's style[ _-]+of and inspired[ _-]+by.
	styleReference = either(joined(false, word("style"), word("of")), joined(false, word("inspired"), word("by")))
	// photoDenied is 474's (?:not|no|without|avoid|exclude|excluding|non)[ _-]*(?:an?[ _-]+)?(?:photo(?:graphs?|graphy|graphic)?
	// |photoReal|hyperReal|realistic).
	photoDenied = then(words("not", "no", "without", "avoid", "exclude", "excluding", "non"), run(joiner, 0),
		maybe(then(words("an", "a"), oneJoiner)),
		either(words("photo", "photographs", "photograph", "photography", "photographic"), photoReal, hyperReal, word("realistic")))
)

// anyAt is whether t reaches an end that after accepts, starting at a position before accepts.
func anyAt(text string, before func(int) bool, t term, after func(int) bool) bool {
	for at := 0; at <= len(text); at++ {
		if !before(at) {
			continue
		}
		if slices.ContainsFunc(t(text, at), after) {
			return true
		}
	}
	return false
}

// photoSource is 474: whether the source idea names a photograph (a photographic word, or a realistic portrait,
// between word boundaries) and names no other medium (between no letters or digits), no @handle, no style of or
// inspired by, and no denial of a photograph.
func photoSource(text string) bool {
	boundary := func(at int) bool {
		before := at > 0 && asciiWord(text[at-1])
		after := at < len(text) && asciiWord(text[at])
		return before != after
	}
	notAfterAlnum := func(at int) bool { return at == 0 || !asciiAlnum(text[at-1]) }
	notBeforeAlnum := func(at int) bool { return at == len(text) || !asciiAlnum(text[at]) }
	handle := func(text string, at int) []int {
		if at+1 < len(text) && text[at] == '@' && asciiWord(text[at+1]) {
			return []int{at + 2}
		}
		return nil
	}
	always := func(int) bool { return true }
	names := anyAt(text, boundary, photoNamed, boundary) || anyAt(text, boundary, realisticPortrait, boundary)
	other := anyAt(text, notAfterAlnum, otherMedium, notBeforeAlnum) || anyAt(text, always, handle, always) ||
		anyAt(text, boundary, styleReference, boundary)
	return names && !other && !anyAt(text, notAfterAlnum, photoDenied, boundary)
}

var (
	// photoGuardKind and guardKind are the kinds a guarded medium may name after it, [ _-]+ before each, at most twice:
	// 477's, and 478's more.
	photoGuardKind = words("style", "like", "aesthetics", "aesthetic", "look", "rendereds", "rendered", "renderings", "rendering", "renders", "render")
	guardKind      = either(photoGuardKind, words("artwork", "art", "illustrations", "illustration", "paintings", "painting"))
	// photoTerm is the guards' photographic medium: photo(?:s|graphs?|graphy|graphic)?, photoReal, hyperReal, realism,
	// realistic and live[ _-]?action.
	photoTerm = either(words("photos", "photographs", "photograph", "photography", "photographic", "photo"), photoReal, hyperReal,
		words("realism", "realistic"), joined(true, word("live"), word("action")))
	// artTerm is the protected guard's other media.
	artTerm = either(words("anime", "manga", "cartoons", "cartoon", "cartoonish", "illustrations", "illustration", "illustrated", "paintings",
		"painting", "painterly"),
		joined(false, words("digital", "traditional"), words("art", "paintings", "painting", "illustrations", "illustration")),
		joined(true, word("2"), word("d")), joined(true, word("3"), word("d")), word("cgi"),
		joined(true, word("cel"), then(word("shad"), words("ed", "ing"))),
		either(words("comics", "comic"), joined(false, word("comic"), word("book"))),
		words("drawings", "drawing"), joined(false, word("hand"), word("drawn")), words("sketches", "sketch"),
		words("watercolour", "watercolor"), joined(false, word("oil"), words("paintings", "painting")), joined(false, word("pixel"), word("art")))
	// skinTerm is the protected guard's skin artefacts.
	skinTerm = either(joined(false, words("plastic", "waxy", "airbrushed", "poreless"), word("skin")),
		joined(false, word("excessive"), word("skin"), words("smoothing", "retouching")),
		joined(false, word("artificial"), word("skin"), word("texture")),
		joined(false, joined(true, word("over"), word("smoothed")), word("skin")))
	// photoGuardTerm is 477's medium, and protectedGuardTerm 478's.
	photoGuardTerm     = then(photoTerm, times(then(oneJoiner, photoGuardKind), 0, 2))
	protectedGuardTerm = either(then(photoTerm, times(then(oneJoiner, guardKind), 0, 2)), then(artTerm, times(then(oneJoiner, guardKind), 0, 2)), skinTerm)
	// guardWeight is a guarded item's weight, :[+-]?(?:\d+(?:\.\d*)?|\.\d+), whitespace after the colon.
	digits      = run(func(b byte) bool { return b >= '0' && b <= '9' }, 1)
	guardWeight = then(word(":"), spaces, maybe(either(word("+"), word("-"))),
		either(then(digits, maybe(then(word("."), run(func(b byte) bool { return b >= '0' && b <= '9' }, 0)))), then(word("."), digits)))
	openers = run(func(b byte) bool { return b == '[' || b == '(' }, 0)
	closers = run(func(b byte) bool { return b == ']' || b == ')' }, 0)
)

// guarded is 477 and 478: each comma-separated item that is, whitespace and brackets about it, a medium the guard
// names, optionally weighted, removed with the comma before it (or, the first, from the text's start).
func guarded(text string, medium term) string {
	item := then(spaces, openers, spaces, medium, spaces, maybe(guardWeight), spaces, closers, spaces)
	var kept strings.Builder
	copied, at := 0, 0
	for at <= len(text) {
		start := at
		body := at
		if at < len(text) && text[at] == ',' {
			body = at + 1
		} else if at != 0 {
			at++
			continue
		}
		next := strings.IndexByte(text[body:], ',')
		end := len(text)
		if next >= 0 {
			end = body + next
		}
		reached := slices.Contains(item(text, body), end)
		if !reached {
			at++
			continue
		}
		kept.WriteString(text[copied:start])
		copied, at = end, end
		if end == len(text) {
			break
		}
	}
	kept.WriteString(text[copied:])
	return kept.String()
}

// guardPhoto is 477, which keeps every medium but a photograph's out of a photograph's negative.
func guardPhoto(text string) string { return guarded(text, photoGuardTerm) }

// guardProtected is 478, which keeps every medium and skin artefact out of any other picture's negative.
func guardProtected(text string) string { return guarded(text, protectedGuardTerm) }

// enhancedPositive is the positive mode 2 gives: the first answer where it holds enough words, else the retry; and the
// source idea where that is empty (451, 456, 452, 471).
func enhancedPositive(source, first, retry string) string {
	answer := retry
	if hasEnhancerWords(first) {
		answer = first
	}
	if answer == "" {
		return source
	}
	return answer
}

// enhancedNegative is the negative mode 2 gives of the negative its enhancer wrote: its first phrases, exact duplicates
// removed, the medium guard of a photograph or of any other picture, its separators trimmed; and the user's exclusions,
// or the backup where they are none, where that leaves no negative (472, 473, 474, 477, 478, 480, 469, 470, 461).
func enhancedNegative(written, source, exclusions, backup string) string {
	guard := guardProtected
	if photoSource(source) {
		guard = guardPhoto
	}
	negative := trimSeparators(guard(withoutDuplicates(firstPhrases(written, enhancerPhrases))))
	if !isNoNegative(negative) {
		return negative
	}
	if isNoNegative(exclusions) {
		return backup
	}
	return exclusions
}
