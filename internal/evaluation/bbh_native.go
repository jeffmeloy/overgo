package evaluation

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"overgo/internal/sequencescore"
)

//go:embed testdata/bbh_native_profile.json
var bbhNativeProfileData []byte

type bbhNativeExample struct {
	Input  string `json:"input"`
	Target string `json:"target"`
}

type bbhNativeGroup struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Choices     []string           `json:"choices"`
	Examples    []bbhNativeExample `json:"examples"`
}

var compiledNativeBBH = sync.OnceValues(func() (map[string]bbhNativeGroup, error) {
	return decodeNativeBBH(bbhNativeProfileData)
})

func decodeNativeBBH(data []byte) (map[string]bbhNativeGroup, error) {
	// Frozen lm_eval task declarations include their source hashes and license.
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != "55a59fc23d9098dfe96e203978810891794181de1ac55012dadb234072fd5df7" {
		return nil, errors.New("evaluation: native BBH profile differs")
	}
	var profile struct {
		Groups []bbhNativeGroup `json:"groups"`
	}
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	groups := make(map[string]bbhNativeGroup, len(profile.Groups))
	for _, group := range profile.Groups {
		if _, found := groups[group.Name]; found {
			return nil, errors.New("evaluation: duplicate native BBH task")
		}
		groups[group.Name] = group
	}
	return groups, nil
}

func assembleNativeBBH(cases []storeCase) (any, int, error) {
	groups, err := compiledNativeBBH()
	if err != nil {
		return nil, 0, err
	}
	suite := GroupedChoiceSuite{
		Kind: GroupedChoiceKind, Schema: "lm-eval/leaderboard-bbh/v1.0", Source: "store/bbh",
		Normalization: sequencescore.NormalizationCharacters, TieBreak: TieBreakFirst,
	}
	for _, entry := range cases {
		group, found := groups[entry.subset]
		if !found {
			return nil, 0, fmt.Errorf("evaluation: undeclared native BBH task %q", entry.subset)
		}
		input, err := caseString(entry.fields, "input")
		if err != nil {
			return nil, 0, err
		}
		target, err := caseString(entry.fields, "target")
		if err != nil {
			return nil, 0, err
		}
		answer := slices.Index(group.Choices, target)
		if answer < 0 {
			return nil, 0, fmt.Errorf("evaluation: undeclared native BBH target %q", target)
		}
		var prompt strings.Builder
		prompt.WriteString(group.Description)
		for _, example := range group.Examples {
			// Native first_n removes an example equal to the entire input document.
			if len(entry.fields) == 2 && example.Input == input && example.Target == target {
				continue
			}
			prompt.WriteString("Q: " + example.Input + "\nA: " + example.Target + "\n\n")
		}
		prompt.WriteString("Q: " + input + "\nA:")
		candidates := make([]string, len(group.Choices))
		characters := make([]uint64, len(group.Choices))
		for index, choice := range group.Choices {
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
