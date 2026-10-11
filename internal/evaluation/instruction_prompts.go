package evaluation

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// InstructionPrompts is a set of prompts with the rules each asks for, as a
// suite of instruction-rules cases states them, compiled: the cases a reward
// run samples answers to and the scorer it scores them by. The scorer is the
// one the instruction-rules evaluation scores by, the same code reading the
// same rules, so a reward and a verdict are counted alike.
type InstructionPrompts struct {
	plan instructionRulesPlan
}

// ParseInstructionPrompts reads a suite of instruction-rules cases.
func ParseInstructionPrompts(data []byte) (InstructionPrompts, error) {
	var suite instructionRulesSuite
	if err := strictjson.DecodeBytes(data, &suite); err != nil {
		return InstructionPrompts{}, fmt.Errorf("evaluation: instruction prompts: %w", err)
	}
	plan, err := compileInstructionRules(suite)
	return InstructionPrompts{plan: plan}, err
}

// Names lists the prompts' names in the suite's order.
func (prompts InstructionPrompts) Names() []string {
	names := make([]string, len(prompts.plan.suite.Cases))
	for index, asked := range prompts.plan.suite.Cases {
		names[index] = asked.Name
	}
	return names
}

// Cases lists the prompts as the cases an instruction acceptance is fixed
// from: each prompt's name as its record, the names of its rules, and a
// digest of what it asks, its prompt, its output limit and its rules with
// their arguments, which is the same wherever the case is registered and
// under whatever name.
func (prompts InstructionPrompts) Cases() []InstructionCase {
	cases := make([]InstructionCase, len(prompts.plan.suite.Cases))
	for index, asked := range prompts.plan.suite.Cases {
		cases[index].Record = asked.Name
		for _, rule := range asked.Rules {
			cases[index].Rules = append(cases[index].Rules, rule.Name)
		}
		asked.Name = ""
		// A case that was read marshals.
		content, _ := json.Marshal(asked)
		cases[index].Content = fmt.Sprintf("%x", sha256.Sum256(content))
	}
	return cases
}

// Suite writes the prompts named, in the order named, as a suite of their
// own: the cases a session over those prompts alone reads, which
// ParseInstructionPrompts reads back. A name the prompts do not hold is
// refused.
func (prompts InstructionPrompts) Suite(names []string) ([]byte, error) {
	suite := prompts.plan.suite
	suite.Cases = make([]instructionRulesCase, len(names))
	for index, name := range names {
		at := slices.IndexFunc(prompts.plan.suite.Cases, func(asked instructionRulesCase) bool { return asked.Name == name })
		if at < 0 {
			return nil, fmt.Errorf("evaluation: no instruction prompt is named %q", name)
		}
		suite.Cases[index] = prompts.plan.suite.Cases[at]
	}
	return json.Marshal(suite)
}

// Evaluator names the scorer the prompts' answers are scored by, as a reward
// group names who gave its rewards.
func (prompts InstructionPrompts) Evaluator() (artifact.ID, error) {
	return artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/evaluator/"+InstructionScorer))
}

// Asked is the text of the prompt named and the most tokens an answer to it
// may run to; found says the suite holds it.
func (prompts InstructionPrompts) Asked(name string) (prompt string, limit int, found bool) {
	for _, asked := range prompts.plan.suite.Cases {
		if asked.Name == name {
			return asked.Prompt, asked.MaxTokens, true
		}
	}
	return "", 0, false
}

// Passed is how many of the named prompt's rules an answer passes, strictly,
// and how many rules it has.
func (prompts InstructionPrompts) Passed(name, answer string) (passed, rules uint64, err error) {
	for index, asked := range prompts.plan.suite.Cases {
		if asked.Name != name {
			continue
		}
		strict, _, err := evaluateInstructionViews(answer, prompts.plan.rules[index])
		for _, ok := range strict {
			if ok {
				passed++
			}
		}
		return passed, uint64(len(strict)), err
	}
	return 0, 0, fmt.Errorf("evaluation: no instruction prompt is named %q", name)
}
