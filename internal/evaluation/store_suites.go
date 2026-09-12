package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/sequencescore"
)

// Derived-suite generation bounds, shared by every family: generation
// cases stop at the same output budgets lm_eval grants them, so scores
// compare across models on identical envelopes.
const (
	mathDerivedMaxTokens = 512
	// lm_eval 0.4.9.1, IFEval task v4.0: generation_kwargs.max_gen_toks.
	ifevalDerivedMaxTokens = 1280
)

// storeCase is one benchmark record joined with the catalog entry it
// came from: the entry name carries family/subset/split, the fields
// carry the case.
type storeCase struct {
	entry      string
	subset     string
	ordinal    int
	fields     map[string]json.RawMessage
	parameters map[string]ifevalParameterBinding
}

// DeriveStoreSuites compiles evaluation suites from the store's active
// benchmark catalog: every imported lm_eval dataset becomes cases of
// the suite kind its family was ported to, mirroring the conventions
// the port's own fixtures pin. No suite files exist anywhere -- the
// store is the configuration. Cases a family cannot express exactly
// (an IFEval instruction outside the rule vocabulary) are skipped and
// counted, never guessed at.
func DeriveStoreSuites(
	ctx context.Context,
	reader artifact.Reader,
	authorities ExactAuthorities,
) ([]CompiledSuite, map[string]int, error) {
	return deriveStoreSuites(ctx, reader, authorities, "")
}

// DeriveStoreSuiteFamily compiles one family's suite alone: the records
// of every other family are neither read nor compiled, so a worker asked
// for a hundred-case family does not spend minutes assembling the whole
// catalog first. An empty family derives every suite.
func DeriveStoreSuiteFamily(
	ctx context.Context,
	reader artifact.Reader,
	authorities ExactAuthorities,
	family string,
) ([]CompiledSuite, map[string]int, error) {
	return deriveStoreSuites(ctx, reader, authorities, family)
}

func deriveStoreSuites(
	ctx context.Context,
	reader artifact.Reader,
	authorities ExactAuthorities,
	wanted string,
) ([]CompiledSuite, map[string]int, error) {
	id, bound, err := artifact.ResolveAlias(ctx, reader, benchmarkCatalogAlias)
	if err != nil || !bound {
		return nil, nil, errors.Join(err, errors.New("evaluation: active benchmark catalog is absent"))
	}
	catalog, found, err := benchmarkCatalogCodec.Read(ctx, reader, id)
	if err != nil || !found {
		return nil, nil, errors.Join(err, errors.New("evaluation: active benchmark catalog is unreadable"))
	}
	families := map[string][]storeCase{}
	for _, entry := range catalog.Entries {
		imported, found, err := dataset.ReadBenchmarkImport(ctx, reader, entry.Dataset)
		if err != nil || !found {
			return nil, nil, errors.Join(err, fmt.Errorf("evaluation: benchmark import %s is unreadable", entry.Name))
		}
		family, recognized := familyForConversion(imported.Spec.Conversion)
		if !recognized || wanted != "" && family.family != wanted {
			continue
		}
		subset := entry.Name
		if parts := strings.Split(entry.Name, "/"); len(parts) >= 2 {
			subset = parts[1]
			if family.family == "musr" {
				subset = imported.Spec.Split
			}
		}
		var parameters map[artifact.ID]map[string]ifevalParameterBinding
		if entry.Parameters.Kind() != artifact.KindInvalid {
			parameters, err = readIFEvalParameters(ctx, reader, imported, entry.Parameters)
			if err != nil {
				return nil, nil, err
			}
		}
		for ordinal, recordID := range imported.Records {
			record, found, err := dataset.ReadBenchmarkRecord(ctx, reader, recordID)
			if err != nil || !found {
				return nil, nil, errors.Join(err, fmt.Errorf("evaluation: benchmark record %d of %s is unreadable", ordinal, entry.Name))
			}
			fields := make(map[string]json.RawMessage, len(record.Fields))
			for _, field := range record.Fields {
				fields[field.Name] = field.Value
			}
			families[family.family] = append(families[family.family], storeCase{
				entry: entry.Name, subset: subset, ordinal: ordinal, fields: fields, parameters: parameters[recordID],
			})
		}
	}
	if len(families) == 0 {
		if wanted != "" {
			return nil, nil, fmt.Errorf("evaluation: the catalog holds no %s family", wanted)
		}
		return nil, nil, errors.New("evaluation: the catalog holds no recognized benchmark families")
	}
	assemblers := map[string]func([]storeCase) (any, int, error){
		"mmlu":     assembleMMLUSuite,
		"mmlu-pro": assembleMMLUProSuite,
		"bbh": func(cases []storeCase) (any, int, error) {
			return assembleChoiceGroups("bbh", "lm-eval/leaderboard-bbh/v1.0", cases)
		},
		"musr":   assembleMuSRSuite,
		"dna":    assembleDNASuite,
		"math":   assembleMATHSuite,
		"ifeval": assembleIFEvalSuite,
	}
	names := slices.Sorted(maps.Keys(families))
	suites := make([]CompiledSuite, 0, len(names))
	skipped := map[string]int{}
	for _, name := range names {
		assemble := assemblers[name]
		source, dropped, err := assemble(families[name])
		if err != nil {
			return nil, nil, fmt.Errorf("evaluation: assemble %s suite: %w", name, err)
		}
		if dropped > 0 {
			skipped[name] = dropped
		}
		data, err := json.Marshal(source)
		if err != nil {
			return nil, nil, err
		}
		compiled, err := CompileSuite(data, authorities)
		if err != nil {
			return nil, nil, fmt.Errorf("evaluation: compile derived %s suite: %w", name, err)
		}
		suites = append(suites, compiled)
	}
	return suites, skipped, nil
}

func caseString(fields map[string]json.RawMessage, name string) (string, error) {
	var value string
	if err := json.Unmarshal(fields[name], &value); err != nil {
		return "", fmt.Errorf("field %q: %w", name, err)
	}
	return value, nil
}

func caseInt(fields map[string]json.RawMessage, name string) (int, error) {
	var value int
	if err := json.Unmarshal(fields[name], &value); err != nil {
		return 0, fmt.Errorf("field %q: %w", name, err)
	}
	return value, nil
}

func caseStrings(fields map[string]json.RawMessage, name string) ([]string, error) {
	var value []string
	if err := json.Unmarshal(fields[name], &value); err != nil {
		return nil, fmt.Errorf("field %q: %w", name, err)
	}
	return value, nil
}

// choiceLetters yields A.. labels for one case's option count.
var choiceLetters = []string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

// assembleMMLUSuite renders MMLU records in the lm_eval convention:
// the lettered options follow the question, the model continues with
// the answer letter, and log-likelihoods pick the choice.
func assembleMMLUSuite(cases []storeCase) (any, int, error) {
	suite := MultipleChoiceSuite{
		Kind: MultipleChoiceKind, Schema: "lm-eval/mmlu/v1.0", Source: "store/mmlu",
		Normalization: sequencescore.NormalizationSum, Aggregation: AggregationAccuracy,
	}
	for _, entry := range cases {
		question, err := caseString(entry.fields, "question")
		if err != nil {
			return nil, 0, err
		}
		choices, err := caseStrings(entry.fields, "choices")
		if err != nil {
			return nil, 0, err
		}
		answer, err := caseInt(entry.fields, "answer")
		if err != nil {
			return nil, 0, err
		}
		if len(choices) > len(choiceLetters) || answer < 0 || answer >= len(choices) {
			return nil, 0, fmt.Errorf("case %s/%d options are outside the letter vocabulary", entry.entry, entry.ordinal)
		}
		prompt := strings.Builder{}
		prompt.WriteString(question)
		prompt.WriteString("\n")
		candidates := make([]string, len(choices))
		for index, choice := range choices {
			fmt.Fprintf(&prompt, "%s. %s\n", choiceLetters[index], choice)
			candidates[index] = " " + choiceLetters[index]
		}
		prompt.WriteString("Answer:")
		suite.Cases = append(suite.Cases, MultipleChoiceCase{
			Name: fmt.Sprintf("%s/%d", entry.entry, entry.ordinal), Prompt: prompt.String(),
			Candidates: candidates, Answer: answer,
		})
	}
	return suite, 0, nil
}

func assembleMMLUProSuite(cases []storeCase) (any, int, error) {
	suite := MMLUProSuite{
		Kind: MMLUProKind, Schema: "lm-eval/leaderboard-mmlu-pro/v0.1", Source: "store/mmlu-pro",
		Labels: choiceLetters, Normalization: sequencescore.NormalizationSum,
	}
	for _, entry := range cases {
		question, err := caseString(entry.fields, "question")
		if err != nil {
			return nil, 0, err
		}
		options, err := caseStrings(entry.fields, "options")
		if err != nil {
			return nil, 0, err
		}
		answer, err := caseInt(entry.fields, "answer_index")
		if err != nil {
			return nil, 0, err
		}
		category, err := caseString(entry.fields, "category")
		if err != nil {
			return nil, 0, err
		}
		if len(options) > len(choiceLetters) || answer < 0 || answer >= len(options) {
			return nil, 0, fmt.Errorf("case %s/%d options are outside the label vocabulary", entry.entry, entry.ordinal)
		}
		suite.Cases = append(suite.Cases, MMLUProCase{
			Name: fmt.Sprintf("%s/%d", entry.entry, entry.ordinal), Category: category,
			Question: question, Options: options, Answer: answer,
		})
	}
	return suite, 0, nil
}

// targetDelimiter separates a prompt that ends in its answer cue ("A:",
// "Answer:") from each scored candidate: lm-eval's target_delimiter, so a
// candidate is scored as the word the model would emit after the cue, not
// glued to the colon.
const targetDelimiter = " "

// assembleChoiceGroups renders extractive-answer records (BBH) as one
// grouped-choice suite: each task is a group, its candidate space is
// the distinct targets the task actually uses, and the recorded target
// picks the answer -- the leaderboard's per-task option-set scoring.
func assembleChoiceGroups(family, schema string, cases []storeCase) (any, int, error) {
	targetsByGroup := map[string][]string{}
	seen := map[string]map[string]int{}
	for _, entry := range cases {
		target, err := caseString(entry.fields, "target")
		if err != nil {
			return nil, 0, err
		}
		if seen[entry.subset] == nil {
			seen[entry.subset] = map[string]int{}
		}
		if _, exists := seen[entry.subset][target]; !exists {
			seen[entry.subset][target] = 0
			targetsByGroup[entry.subset] = append(targetsByGroup[entry.subset], target)
		}
	}
	candidatesByGroup := map[string][]string{}
	for group := range targetsByGroup {
		slices.Sort(targetsByGroup[group])
		candidatesByGroup[group] = make([]string, len(targetsByGroup[group]))
		for index, target := range targetsByGroup[group] {
			seen[group][target] = index
			candidatesByGroup[group][index] = targetDelimiter + target
		}
	}
	suite := GroupedChoiceSuite{
		Kind: GroupedChoiceKind, Schema: schema, Source: "store/" + family,
		Normalization: sequencescore.NormalizationMean,
	}
	for _, entry := range cases {
		input, err := caseString(entry.fields, "input")
		if err != nil {
			return nil, 0, err
		}
		target, err := caseString(entry.fields, "target")
		if err != nil {
			return nil, 0, err
		}
		suite.Cases = append(suite.Cases, DemonstratedChoice{
			Name: fmt.Sprintf("%s/%d", entry.entry, entry.ordinal), Group: entry.subset,
			Prompt:     "Q: " + input + "\nA:",
			Candidates: candidatesByGroup[entry.subset], Answer: seen[entry.subset][target],
		})
	}
	return suite, 0, nil
}

func assembleMuSRSuite(cases []storeCase) (any, int, error) {
	suite := GroupedChoiceSuite{
		Kind: GroupedChoiceKind, Schema: "lm-eval/leaderboard-musr/v1.0", Source: "store/musr",
		Normalization: sequencescore.NormalizationCharacters, TieBreak: TieBreakFirst,
	}
	for _, entry := range cases {
		narrative, err := caseString(entry.fields, "narrative")
		if err != nil {
			return nil, 0, err
		}
		question, err := caseString(entry.fields, "question")
		if err != nil {
			return nil, 0, err
		}
		encoded, err := caseString(entry.fields, "choices")
		if err != nil {
			return nil, 0, err
		}
		choices, err := parsePythonStringList(encoded)
		if err != nil {
			return nil, 0, fmt.Errorf("case %s/%d choices: %w", entry.entry, entry.ordinal, err)
		}
		answer, err := caseInt(entry.fields, "answer_index")
		if err != nil {
			return nil, 0, err
		}
		if answer < 0 || answer >= len(choices) {
			return nil, 0, fmt.Errorf("case %s/%d answer is outside its choices", entry.entry, entry.ordinal)
		}
		prompt := strings.Builder{}
		prompt.WriteString(narrative)
		prompt.WriteString("\n\n")
		prompt.WriteString(question)
		prompt.WriteString("\n\n")
		for index, choice := range choices {
			fmt.Fprintf(&prompt, "%d - %s\n", index+1, choice)
		}
		prompt.WriteString("Answer:")
		candidates := make([]string, len(choices))
		characters := make([]uint64, len(choices))
		for index, choice := range choices {
			candidates[index] = targetDelimiter + choice
			characters[index] = uint64(utf8.RuneCountInString(choice))
		}
		suite.Cases = append(suite.Cases, DemonstratedChoice{
			Name: fmt.Sprintf("%s/%d", entry.entry, entry.ordinal), Group: entry.subset,
			Prompt: prompt.String(), Candidates: candidates, CandidateCharacters: characters, Answer: answer,
		})
	}
	return suite, 0, nil
}

func assembleMATHSuite(cases []storeCase) (any, int, error) {
	suite := StructuredGeneratedSuite{
		Kind: StructuredGeneratedKind, Schema: "lm-eval/hendrycks-math/v1.0", Source: "store/math",
		Extractor: ExtractorDollarSpan, Equivalence: EquivalenceLatexSurface,
	}
	dropped := 0
	for _, entry := range cases {
		problem, err := caseString(entry.fields, "problem")
		if err != nil {
			return nil, 0, err
		}
		solution, err := caseString(entry.fields, "solution")
		if err != nil {
			return nil, 0, err
		}
		answer, found := extractBoxed(solution)
		if !found {
			dropped++
			continue
		}
		suite.Cases = append(suite.Cases, StructuredGeneratedCase{
			Name: fmt.Sprintf("%s/%d", entry.entry, entry.ordinal), Group: entry.subset,
			Prompt:    "Problem: " + problem + "\nAnswer:",
			MaxTokens: mathDerivedMaxTokens, Answers: []string{answer},
		})
	}
	if len(suite.Cases) == 0 {
		return nil, dropped, errors.New("no MATH solution carried a boxed answer")
	}
	return suite, dropped, nil
}

// extractBoxed returns the balanced content of the last \boxed{...}
// span -- the MATH corpus convention for the final answer.
func extractBoxed(solution string) (string, bool) {
	const marker = `\boxed{`
	start := strings.LastIndex(solution, marker)
	if start < 0 {
		return "", false
	}
	depth := 1
	for index := start + len(marker); index < len(solution); index++ {
		switch solution[index] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return solution[start+len(marker) : index], true
			}
		}
	}
	return "", false
}

// ifevalRuleFor maps one lm_eval instruction id and its kwargs onto
// the port's rule vocabulary. Instructions outside the vocabulary
// report unmapped so their cases are skipped and counted -- a rule
// that cannot be checked exactly is not approximated.
func ifevalRuleFor(id string, kwargs map[string]json.RawMessage) ([]InstructionRule, bool) {
	stringsOf := func(name string) ([]string, bool) {
		values, err := caseStrings(kwargs, name)
		return values, err == nil && len(values) > 0
	}
	switch id {
	case "keywords:existence":
		values, ok := stringsOf("keywords")
		if !ok {
			return nil, false
		}
		return []InstructionRule{{Name: id, Kind: ifevalKeywordsRule, Values: values}}, true
	case "keywords:forbidden_words":
		values, ok := stringsOf("forbidden_words")
		if !ok {
			return nil, false
		}
		return []InstructionRule{{Name: id, Kind: ifevalForbiddenRule, Values: values}}, true
	case "change_case:english_capital":
		return []InstructionRule{{Name: id, Kind: RuleUppercase}}, true
	case "change_case:english_lowercase":
		return []InstructionRule{{Name: id, Kind: RuleLowercase}}, true
	case "punctuation:no_comma":
		return []InstructionRule{{Name: id, Kind: RuleNoComma}}, true
	case "startend:quotation":
		// Python str.strip whitespace; the native check requires two quotes.
		const space = ifevalSpaceClass + "*"
		return []InstructionRule{{Name: id, Kind: RuleRegex, Values: []string{`(?s)^` + space + `".*"` + space + `$`}}}, true
	case "startend:end_checker":
		var phrase string
		if err := json.Unmarshal(kwargs["end_phrase"], &phrase); err != nil || phrase == "" {
			return nil, false
		}
		return []InstructionRule{{Name: id, Kind: ifevalEndRule, Values: []string{phrase}}}, true
	case "detectable_format:json_format":
		return []InstructionRule{{Name: id, Kind: ifevalJSONRule}}, true
	default:
		return ifevalStructureFor(id, kwargs)
	}
}

func assembleIFEvalSuite(cases []storeCase) (any, int, error) {
	suite := InstructionRulesSuite{
		Kind: InstructionRulesKind, Schema: "lm-eval/ifeval/v4.0", Source: "store/ifeval",
	}
	dropped := 0
	for _, entry := range cases {
		prompt, err := caseString(entry.fields, "prompt")
		if err != nil {
			return nil, 0, err
		}
		ids, err := caseStrings(entry.fields, "instruction_id_list")
		if err != nil {
			return nil, 0, err
		}
		var kwargsList []map[string]json.RawMessage
		if raw, present := entry.fields["kwargs"]; present {
			if err := json.Unmarshal(raw, &kwargsList); err != nil {
				return nil, 0, fmt.Errorf("case %s/%d kwargs: %w", entry.entry, entry.ordinal, err)
			}
		}
		rules := make([]InstructionRule, 0, len(ids))
		mapped := true
		for index, id := range ids {
			kwargs := map[string]json.RawMessage{}
			if index < len(kwargsList) && kwargsList[index] != nil {
				kwargs = kwargsList[index]
			}
			var binding *ifevalParameterBinding
			if bound, present := entry.parameters[id]; present {
				binding = &bound
			}
			ruleSet, ok := ifevalRuleWithParameters(id, kwargs, binding)
			if !ok {
				mapped = false
				break
			}
			rules = append(rules, ruleSet...)
		}
		if !mapped || len(rules) == 0 {
			dropped++
			continue
		}
		suite.Cases = append(suite.Cases, InstructionRulesCase{
			Name: fmt.Sprintf("%s/%d", entry.entry, entry.ordinal), Prompt: prompt,
			MaxTokens: ifevalDerivedMaxTokens, Rules: rules,
		})
	}
	if len(suite.Cases) == 0 {
		return nil, dropped, errors.New("no IFEval case mapped onto the rule vocabulary")
	}
	return suite, dropped, nil
}

// parsePythonStringList decodes the Python list repr MuSR stores its
// choices as: bracketed, comma-separated, single- or double-quoted
// strings with backslash escapes.
func parsePythonStringList(encoded string) ([]string, error) {
	text := strings.TrimSpace(encoded)
	if len(text) < 2 || text[0] != '[' || text[len(text)-1] != ']' {
		return nil, errors.New("not a bracketed list")
	}
	var values []string
	index := 1
	for index < len(text)-1 {
		for index < len(text)-1 && (text[index] == ' ' || text[index] == ',') {
			index++
		}
		if index >= len(text)-1 {
			break
		}
		quote := text[index]
		if quote != '\'' && quote != '"' {
			return nil, errors.New("list element is not quoted")
		}
		index++
		value := strings.Builder{}
		for index < len(text)-1 && text[index] != quote {
			if text[index] == '\\' && index+1 < len(text)-1 {
				index++
			}
			value.WriteByte(text[index])
			index++
		}
		if index >= len(text)-1 {
			return nil, errors.New("unterminated list element")
		}
		index++
		values = append(values, value.String())
	}
	if len(values) == 0 {
		return nil, errors.New("empty list")
	}
	return values, nil
}

// familyForConversion reports the declared cache family carrying the
// conversion label, so suite derivation binds catalog entries back to
// their field vocabulary.
func familyForConversion(conversion string) (hfCacheFamily, bool) {
	for _, family := range hfCacheFamilies {
		if family.conversion == strings.TrimSpace(conversion) {
			return family, true
		}
	}
	return hfCacheFamily{}, false
}
