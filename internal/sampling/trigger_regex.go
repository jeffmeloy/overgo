package sampling

import (
	"fmt"
	"time"

	"github.com/dlclark/regexp2/v2"
)

// regexp2 is overgo's one justified non-stdlib dep (pure Go, no cgo, leaf).
// Retained: GBNF triggers are ECMAScript for std::regex parity — lookaround +
// backreferences (tested: TestLazyGBNFECMAScriptTriggerFeatures) that stdlib
// RE2 cannot express. ReDoS bounded by backtrack cap + MatchTimeout + size/count
// limits below. See docs/stdlib_only_plan.md §4.

const (
	maxGBNFTriggerPatternBytes = 4096
	maxGBNFTriggerBacktrack    = 32768
	maxGBNFTriggerMatchTime    = 50 * time.Millisecond
)

type gbnfTriggerRegex struct {
	compiled *regexp2.Regexp
}

func compileGBNFTriggerRegex(source string) (*gbnfTriggerRegex, error) {
	if len(source) == 0 {
		return nil, fmt.Errorf("pattern is empty")
	}
	if len(source) > maxGBNFTriggerPatternBytes {
		return nil, fmt.Errorf(
			"pattern exceeds %d bytes",
			maxGBNFTriggerPatternBytes,
		)
	}
	compiled, err := regexp2.Compile(
		source,
		regexp2.ECMAScript|regexp2.Unicode,
		regexp2.OptionMaxBacktrackingStackSize(maxGBNFTriggerBacktrack),
	)
	if err != nil {
		return nil, err
	}
	compiled.MatchTimeout = maxGBNFTriggerMatchTime
	return &gbnfTriggerRegex{compiled: compiled}, nil
}

func (r *gbnfTriggerRegex) findStart(input []byte) (int, bool, error) {
	match, err := r.compiled.FindStringMatch(string(input))
	if err != nil {
		return 0, false, err
	}
	if match == nil {
		return 0, false, nil
	}
	start, _ := match.ByteRange()
	groups := match.Groups()
	for index := 1; index < len(groups); index++ {
		groupStart, groupLength := groups[index].ByteRange()
		if groupLength > 0 {
			return groupStart, true, nil
		}
	}
	return start, true, nil
}
