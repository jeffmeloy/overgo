package inferencesurface

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// TestSurfaceMoveRecordDefinesClaimDependencies holds the record of what past commits expired and reached to being whole.
//   - range = a head, an oldest commit and a base by their full names, and one row for each commit counted
//   - row = a commit's kind, measurement run, keys moved, regions, narrower keys that still move, identities moved, lanes run and seconds
//   - narrower key moves => the key it narrows moves, in every row
//   - figure = what the rows give, for every figure the record states and no figure besides
//   - package = the lane runs reached through it, and the runs and lane seconds the rows say a stop there cuts
//   - check = the probe's result beside the code's own, equal in every part
//   - claim kind = an answer, a score, a timing and an activation, each with its dependencies named
//   - dependency = a statement with places in the tree, each a file that holds the name the place gives
//   - declared set = the set the code declares, name for name
//   - closing = the largest cost, the candidate and the boundary that avoid most, as the figures give them
//   - the record refuses a key stated twice, because a later line would replace an earlier one unseen
func TestSurfaceMoveRecordDefinesClaimDependencies(t *testing.T) {
	t.Parallel()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(root, "docs", "verification", "surface-moves.txt"))
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]string{}
	var commits, stated []string
	const commitsPart, figuresPart = "== commits ==", "== figures =="
	var part string
	for line := range strings.Lines(string(raw)) {
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, "== ") {
			part = line
			continue
		}
		if part == commitsPart && line != "" {
			commits = append(commits, line)
			continue
		}
		key, value, found := strings.Cut(line, ": ")
		if !found || strings.Contains(key, " ") {
			continue
		}
		if _, twice := fields[key]; twice {
			t.Fatalf("the record states %s twice", key)
		}
		fields[key] = value
		if part == figuresPart {
			stated = append(stated, key)
		}
	}
	count := func(name string) int {
		t.Helper()
		value, err := strconv.Atoi(fields[name])
		if err != nil || value < 0 {
			t.Fatalf("the record's %s is %q (%v), want a count", name, fields[name], err)
		}
		return value
	}

	// The range: three commits by their full names.
	name := regexp.MustCompile(`^[0-9a-f]{40}$`)
	for _, end := range []string{"range-head", "range-oldest", "range-base"} {
		if !name.MatchString(fields[end]) {
			t.Fatalf("the record's %s is %q, want a commit's full name", end, fields[end])
		}
	}

	// The rows: each is read whole, and every figure is counted from them.
	const (
		onLine, merged, beside = "line", "merge", "side"
		rowWords               = "commit name kind run keys text scope still identities device webui cuda slice seconds"
		textKey, scopeKey      = "T", "E"
		importClasses          = "it"
	)
	lanes := []string{"device", "webui", "cuda"}
	classes := map[byte]string{'m': "module", 'p': "prefix", 'o': "own-test", 'i': "production", 't': "test-import", 'u': "unexplained"}
	textRegions := map[string]string{"r": "runner", "x": "executor", "c": "catalog", "s": "several", "n": "membership"}
	scopeRegions := map[string]string{"e": "evaluation", "c": "command", "o": "opener", "s": "several", "n": "membership"}
	parts := []string{"queue-wait", "timed-measurement-wait", "test", "lanes", "lane-turn-wait", "device-admission", "other"}
	// narrowed names, for each narrower key's letter, the key it narrows and the letters that move whenever it does.
	narrowed := map[byte]struct{ key, with string }{
		'1': {scopeKey, ""}, '2': {scopeKey, ""}, '6': {scopeKey, "12"},
		'3': {textKey, ""}, '4': {textKey, "3"}, '5': {textKey, ""}, '7': {textKey, ""},
	}
	stills := map[string]string{
		"candidate-1-scope-still": "1", "candidate-2-scope-still": "2", "candidate-6-scope-still": "6",
		"candidate-3-text-still": "3", "candidate-4-text-still": "4", "candidate-5-text-still": "5", "candidate-7-text-still": "7",
	}
	// answerKeys names, for each figure of the answer a native text evaluation keeps, the letters any of which moves it.
	answerKeys := map[string]string{
		"answer-key-moved": "TE", "answer-key-still-1": "T1", "answer-key-still-2": "T2", "answer-key-still-3": "3E", "answer-key-still-4": "4E", "answer-key-still-2-4": "24",
	}
	// runKeys names, for each figure counted in measurement runs, the letters any of which moves it.
	runKeys := maps.Clone(answerKeys)
	runKeys["trigger-moved"] = "G"
	const allRuns, liveRuns = "runs-", "runs-live-"
	runs, live := count("runs"), count("runs-live-first")
	if live > runs || count("runs-live") != runs-live {
		t.Fatalf("the record's live period opens at run %d of %d and holds %s runs", live, runs, fields["runs-live"])
	}
	figures := map[string]int{}
	byKind := map[string]int{}
	moved := map[string]map[int]bool{}
	// packageRuns is what the rows give one package, the lane runs reached through it and the runs and lane seconds a stop at it cuts.
	type packageRuns struct{ through, cut, seconds int }
	const packagePrefix = "package-"
	packages := map[int]packageRuns{}
	seen := map[string]bool{}
	for _, line := range commits {
		words := strings.Fields(line)
		if len(words) != len(strings.Fields(rowWords)) || words[0] != "commit" || seen[words[1]] {
			t.Fatalf("the record's row %q is not one commit named once in the form %q", line, rowWords)
		}
		seen[words[1]] = true
		kind, keys, still := words[2], words[4], words[7]
		side := ""
		switch kind {
		case onLine, merged:
		case beside:
			side = "side-"
		default:
			t.Fatalf("the record's row %q is of kind %q, want %s, %s or %s", line, kind, onLine, merged, beside)
		}
		byKind[kind]++
		has := func(letters string) bool { return strings.ContainsAny(keys+still, letters) }
		if strings.Trim(keys, "TESG") != "" && keys != "-" || strings.Trim(still, "1234567") != "" && still != "-" {
			t.Fatalf("the record's row %q names keys %q and narrower keys %q, want letters of TESG and of 1234567", line, keys, still)
		}
		for letter, narrows := range narrowed {
			if !strings.ContainsRune(still, rune(letter)) {
				continue
			}
			for _, needed := range narrows.key + narrows.with {
				if !has(string(needed)) {
					t.Fatalf("the record's row %q moves the narrower key %c and not %c, which it narrows", line, letter, needed)
				}
			}
		}
		// A region is named exactly when its key moved, and a sole region says which narrower keys can still move.
		text, scope := words[5], words[6]
		if (text != "-") != has(textKey) || (scope != "-") != has(scopeKey) || text != "-" && textRegions[text] == "" || scope != "-" && scopeRegions[scope] == "" ||
			text == "c" && has("3") || text == "x" && (!has("3") || has("4")) || text == "r" && !has("4") {
			t.Fatalf("the record's row %q names the regions %q and %q, which its keys %q and narrower keys %q do not bear out", line, text, scope, keys, still)
		}
		var identities [4]int
		for index, held := range strings.Split(words[8], "/") {
			value, err := strconv.Atoi(held)
			if err != nil || index >= len(identities) {
				t.Fatalf("the record's row %q counts identities as %q, want moved/held/moved without the catalog/moved alone", line, words[8])
			}
			identities[index] = value
		}
		if identities[0] > identities[1] || identities[2] > identities[0] || identities[3] > identities[2] || has(textKey) && identities[0] == 0 {
			t.Fatalf("the record's row %q counts identities as %q: want each narrower count within the wider, and the text identity among those moved", line, words[8])
		}
		add := func(name string, by int) { figures[side+name] += by }
		flag := func(held bool) int {
			if held {
				return 1
			}
			return 0
		}
		add("text-moved", flag(has(textKey)))
		add("text-unmoved", flag(!has(textKey)))
		add("scope-moved", flag(has(scopeKey)))
		add("scope-unmoved", flag(!has(scopeKey)))
		for letter, region := range textRegions {
			add("text-moved-"+region, flag(text == letter))
		}
		for letter, region := range scopeRegions {
			add("scope-moved-"+region, flag(scope == letter))
		}
		add("surface-moved", flag(has("S")))
		add("trigger-moved", flag(has("G")))
		add("answer-key-unmoved", flag(!has(answerKeys["answer-key-moved"])))
		for figure, letters := range answerKeys {
			add(figure, flag(has(letters)))
		}
		for figure, letter := range stills {
			add(figure, flag(has(letter)))
		}
		add("identities-all-moved", flag(identities[0] > 0 && identities[0] == identities[1]))
		add("identities-some-moved", flag(identities[0] > 0 && identities[0] < identities[1]))
		add("identities-none-moved", flag(identities[0] == 0))
		add("identity-expiries", identities[0])
		add("identity-commit-pairs", identities[1])
		add("candidate-3-identity-expiries-still", identities[2])
		add("candidate-4-identity-expiries-still", identities[3])
		if (kind == beside) != (words[3] == "-") {
			t.Fatalf("the record's row %q is of kind %q in run %q: want a run for a commit of the line and none for one beside it", line, kind, words[3])
		}
		if kind != beside {
			run, err := strconv.Atoi(words[3])
			if err != nil || run < 0 || run > runs {
				t.Fatalf("the record's row %q is in run %q, want one of the %d runs or the one that has not been made", line, words[3], runs)
			}
			// A run asks again when a commit it measured first moved the key, and the live period is the runs from its first on.
			for figure, letters := range runKeys {
				if !has(letters) || run == runs {
					continue
				}
				for _, period := range []string{allRuns, liveRuns} {
					if period == liveRuns && run < live {
						continue
					}
					if moved[period+figure] == nil {
						moved[period+figure] = map[int]bool{}
					}
					moved[period+figure][run] = true
				}
			}
		}
		for index, lane := range lanes {
			field := words[9+index]
			if field == "-" {
				continue
			}
			first, stops, _ := strings.Cut(field, "/")
			reach, through, _ := strings.Cut(first, "@")
			class, known := classes[reach[0]]
			spent, err := strconv.Atoi(reach[1:])
			imported := strings.ContainsRune(importClasses, rune(reach[0]))
			if !known || err != nil || stops != "" && !imported || through == "" && imported {
				t.Fatalf("the record's row %q runs the %s lane as %q: want a reach class and whole seconds, a package for a reach through imports, and a stop only for such a reach", line, lane, field)
			}
			add("lane-"+lane+"-ran", 1)
			add("lane-"+lane+"-seconds", spent)
			for _, each := range classes {
				add("lane-"+lane+"-reach-"+each, flag(each == class))
			}
			add("boundary-test-imports-reaches", flag(reach[0] == 't'))
			add("boundary-test-imports-seconds", flag(reach[0] == 't')*spent)
			// A package is counted once for each run reached through it, and once with the run's seconds for each run a stop at it cuts.
			for _, numbered := range []struct {
				list string
				cuts bool
			}{{through, false}, {stops, true}} {
				for each := range strings.SplitSeq(numbered.list, ".") {
					if each == "" {
						continue
					}
					number, err := strconv.Atoi(each)
					if err != nil || fields[packagePrefix+each] == "" {
						t.Fatalf("the record's row %q names the package %q, which the record does not list", line, each)
					}
					if kind == beside {
						continue
					}
					held := packages[number]
					if numbered.cuts {
						held.cut, held.seconds = held.cut+1, held.seconds+spent
					} else {
						held.through++
					}
					packages[number] = held
				}
			}
		}
		if words[12] != "0" && words[12] != "1" {
			t.Fatalf("the record's row %q says the slice was measured %q, want 0 or 1", line, words[12])
		}
		add("slice-measured", flag(words[12] == "1"))
		if words[13] == "-" {
			if kind != beside {
				t.Fatalf("the record's row %q is a commit of the line with no seconds", line)
			}
			continue
		}
		spent := strings.Split(words[13], ",")
		if len(spent) != len(parts) {
			t.Fatalf("the record's row %q gives %d parts of its seconds, want %d: %s", line, len(spent), len(parts), strings.Join(parts, ","))
		}
		add("measured-commits", 1)
		for index, part := range parts {
			value, err := strconv.Atoi(spent[index])
			if err != nil || value < 0 {
				t.Fatalf("the record's row %q gives %q seconds of %s, want whole seconds", line, spent[index], part)
			}
			add("seconds-"+part, value)
			add("seconds-total", value)
		}
	}
	// rangeSize is how many commits of the first-parent line the record classifies.
	//   - source = the plan row evidence-identity, which asks for the last 500 commits
	//   - scope = this record
	//   - reconsider when a later measurement replaces the record
	const rangeSize = 500
	if count("commits") != rangeSize || byKind[onLine]+byKind[merged] != rangeSize || count("commits-line") != byKind[onLine] ||
		count("commits-merge") != byKind[merged] || count("commits-side") != byKind[beside] || figures["measured-commits"] != rangeSize {
		t.Fatalf("the record counts %s commits, %s of the line, %s merges and %s beside, and its rows hold %v with %d of the line measured: want %d of the line, each measured",
			fields["commits"], fields["commits-line"], fields["commits-merge"], fields["commits-side"], byKind, figures["measured-commits"], rangeSize)
	}
	for figure := range runKeys {
		for _, period := range []string{allRuns, liveRuns} {
			figures[period+figure] = len(moved[period+figure])
		}
	}

	// The figures: each the record states is what the rows give, and the record states every one of the line's.
	sideFigures := []string{"side-text-moved", "side-scope-moved", "side-answer-key-moved", "side-surface-moved", "side-trigger-moved", "side-identity-expiries",
		"side-seconds-total", "side-measured-commits", "side-lane-device-ran", "side-lane-webui-ran", "side-lane-cuda-ran", "side-slice-measured"}
	const largestCost = "largest-cost"
	for _, figure := range stated {
		if want, counted := figures[figure]; figure != largestCost && (!counted || count(figure) != want) {
			t.Fatalf("the record states %s as %s, and its rows give %d (counted from them %t)", figure, fields[figure], want, counted)
		}
	}
	for figure := range figures {
		if _, given := fields[figure]; !given && (!strings.HasPrefix(figure, "side-") || slices.Contains(sideFigures, figure)) {
			t.Fatalf("the record leaves out %s, which its rows give as %d", figure, figures[figure])
		}
	}
	// The sums close: a key moved or did not, a moved key has one region, and a lane that ran has one reach class.
	if figures["text-moved"]+figures["text-unmoved"] != rangeSize || figures["scope-moved"]+figures["scope-unmoved"] != rangeSize ||
		figures["answer-key-moved"]+figures["answer-key-unmoved"] != rangeSize ||
		figures["identities-all-moved"]+figures["identities-some-moved"]+figures["identities-none-moved"] != rangeSize {
		t.Fatalf("the record's keys do not each divide the %d commits into moved and unmoved: %v", rangeSize, figures)
	}
	regions, reached := map[string]int{}, map[string]int{}
	for _, region := range textRegions {
		regions["text-moved"] += figures["text-moved-"+region]
	}
	for _, region := range scopeRegions {
		regions["scope-moved"] += figures["scope-moved-"+region]
	}
	for _, lane := range lanes {
		for _, class := range classes {
			reached[lane] += figures["lane-"+lane+"-reach-"+class]
		}
		if reached[lane] != figures["lane-"+lane+"-ran"] || figures["lane-"+lane+"-ran"] == 0 || figures["lane-"+lane+"-reach-unexplained"] != 0 {
			t.Fatalf("the record's %s lane ran %d times, by %d reaches classed and %d unexplained: want every run classed", lane, figures["lane-"+lane+"-ran"], reached[lane], figures["lane-"+lane+"-reach-unexplained"])
		}
	}
	if regions["text-moved"] != figures["text-moved"] || regions["scope-moved"] != figures["scope-moved"] || figures["text-moved"] == 0 || figures["scope-moved"] == 0 {
		t.Fatalf("the record's regions hold %v moves of the %d and %d its keys made", regions, figures["text-moved"], figures["scope-moved"])
	}
	// The seconds: the parts are the total, and the largest cost is the largest part.
	var total, most int
	var largest string
	for _, part := range parts {
		total += figures["seconds-"+part]
		if figures["seconds-"+part] > most {
			largest, most = part, figures["seconds-"+part]
		}
	}
	if total != figures["seconds-total"] || total == 0 || fields[largestCost] != largest {
		t.Fatalf("the record's parts hold %d seconds of a total of %d, and it names %q the largest where the parts name %q", total, figures["seconds-total"], fields[largestCost], largest)
	}

	// The packages: each with the lane runs reached through it, and the runs and seconds the rows say a stop there cuts.
	var widest string
	var widestCut packageRuns
	for number := 1; ; number++ {
		listed, given := fields[packagePrefix+strconv.Itoa(number)]
		if !given {
			if len(packages) == 0 || slices.Max(slices.Collect(maps.Keys(packages))) >= number {
				t.Fatalf("the record lists %d packages and its rows name %d, none or one beyond them", number-1, len(packages))
			}
			break
		}
		words := strings.Fields(listed)
		held := packages[number]
		if want := fmt.Sprintf("%d %d %d", held.through, held.cut, held.seconds); len(words) != len(strings.Fields("package through cut seconds")) || strings.Join(words[1:], " ") != want {
			t.Fatalf("the record's package %d is %q, and its rows give it %s", number, listed, want)
		}
		if held.seconds > widestCut.seconds {
			widest, widestCut = words[0], held
		}
	}

	// The checks: the probe's result is the code's own in every part.
	checks := 0
	for key, value := range fields {
		if !strings.HasPrefix(key, "check-") {
			continue
		}
		checks++
		words := strings.Fields(value)
		if len(words) < 3 || words[1] != "of" || words[0] != words[2] || words[0] == "0" {
			t.Fatalf("the record's %s is %q, want the probe and the code equal in every part", key, value)
		}
	}
	if checks == 0 {
		t.Fatal("the record holds no check of the probe against the code")
	}

	// The definitions: each claim kind names its dependencies, and each dependency its places in the tree.
	place := regexp.MustCompile(`^([A-Za-z0-9_./-]+\.go):[0-9]+#([A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)$`)
	sources := map[string]string{}
	named := map[string]bool{}
	for _, kind := range []string{"answer", "score", "timing", "activation"} {
		tokens := strings.Fields(fields["depends-"+kind])
		if len(tokens) == 0 {
			t.Fatalf("the record does not say what the claim kind %s depends on", kind)
		}
		for _, token := range tokens {
			named[token] = true
			if fields["dependency-"+token] == "" {
				t.Fatalf("the record says %s depends on %s and does not define it", kind, token)
			}
		}
	}
	for key, value := range fields {
		token, dependency := strings.CutPrefix(key, "dependency-")
		if _, measuredKey := strings.CutPrefix(key, "key-"); !dependency && !measuredKey {
			continue
		}
		if dependency && !named[token] {
			t.Fatalf("the record defines the dependency %s, which no claim kind names", token)
		}
		statement, places, found := strings.Cut(value, " | ")
		if !found || strings.TrimSpace(statement) == "" || len(strings.Fields(places)) == 0 {
			t.Fatalf("the record's %s is %q, want a statement, a bar and places in the tree", key, value)
		}
		for cited := range strings.FieldsSeq(places) {
			match := place.FindStringSubmatch(cited)
			if match == nil {
				t.Fatalf("the record's %s cites %q, want file.go:line#Name", key, cited)
			}
			source, read := sources[match[1]]
			if !read {
				data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(match[1])))
				if err != nil {
					t.Fatalf("the record's %s cites %s, which the tree does not hold: %v", key, match[1], err)
				}
				source = string(data)
				sources[match[1]] = source
			}
			if !regexp.MustCompile(`\b` + regexp.QuoteMeta(match[3]) + `\b`).MatchString(source) {
				t.Fatalf("the record's %s cites %s in %s, which does not hold that name", key, match[3], match[1])
			}
		}
	}

	// The declared sets: the code's, name for name.
	for key, want := range map[string][]string{
		"set-surface-roots": surfaceRoots, "set-surface-kernels": surfaceKernels, "set-identity-addressed": identityAddressed,
		"set-build-settings": buildSettings, "set-result-settings": resultSettings, "set-libraries": libraries,
		"set-evaluation-binders": evaluationBinders, "set-capability-catalog": {capabilityCatalog},
		"set-capability-executor": {capabilityExecutor}, "set-text-runner": {textRunner},
	} {
		if got := strings.Fields(fields[key]); !slices.Equal(got, want) {
			t.Fatalf("the record declares %s as %q, and the code declares %q", key, got, want)
		}
	}

	// The judgements: one for each narrower key the row asked for and for the lane boundaries, and whose they are.
	for _, key := range []string{"judgement-by", "judgement-candidate-1", "judgement-candidate-2", "judgement-candidate-3", "judgement-candidate-4", "judgement-boundaries"} {
		if strings.TrimSpace(fields[key]) == "" {
			t.Fatalf("the record gives no %s", key)
		}
	}

	// The closing: the candidate and the boundary that avoid most are the figures' own, with what they avoid.
	share := func(part, whole int) string { return fmt.Sprintf("%.2f%%", 100*float64(part)/float64(whole)) }
	best, avoided := "", -1
	for _, candidate := range []string{"1", "2", "3", "4"} {
		if spared := figures["answer-key-moved"] - figures["answer-key-still-"+candidate]; spared > avoided {
			best, avoided = candidate, spared
		}
	}
	for key, want := range map[string]string{
		"closing-largest-cost":                       largest,
		"closing-largest-cost-share":                 share(most, total),
		"closing-key-candidate":                      best,
		"closing-key-candidate-commits-avoided":      strconv.Itoa(avoided),
		"closing-key-candidate-runs-avoided":         strconv.Itoa(figures["runs-answer-key-moved"] - figures["runs-answer-key-still-"+best]),
		"closing-key-candidate-live-runs-avoided":    strconv.Itoa(figures["runs-live-answer-key-moved"] - figures["runs-live-answer-key-still-"+best]),
		"closing-key-candidates-2-4-commits-avoided": strconv.Itoa(figures["answer-key-moved"] - figures["answer-key-still-2-4"]),
		"closing-key-candidates-2-4-runs-avoided":    strconv.Itoa(figures["runs-answer-key-moved"] - figures["runs-answer-key-still-2-4"]),
		"closing-boundary":                           widest,
		"closing-boundary-lane-runs-cut":             strconv.Itoa(widestCut.cut),
		"closing-boundary-lane-seconds-cut":          strconv.Itoa(widestCut.seconds),
		"closing-boundary-share-of-all-seconds":      share(widestCut.seconds, total),
	} {
		if fields[key] != want {
			t.Fatalf("the record closes with %s %q, and its figures give %q", key, fields[key], want)
		}
	}
	for _, key := range []string{"closing-key-candidate-row", "closing-boundary-row"} {
		if answer, _, _ := strings.Cut(fields[key], ","); answer != "yes" && answer != "no" {
			t.Fatalf("the record closes with %s %q, want yes or no and the reason", key, fields[key])
		}
	}
}
