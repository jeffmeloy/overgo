package trainingdata

import (
	"testing"

	"overgo/internal/recipecontract"
)

func TestPreferenceExampleContract(t *testing.T) {
	reference := recordRef{id: "pair", group: "group"}
	processor := compiledProcessor{modalities: map[recipecontract.Modality]bool{recipecontract.ModalityText: true}}
	value := func(role ValueRole) Value {
		return Value{
			Role: role, Modality: recipecontract.ModalityText, Encoding: "tokens",
			Data: []byte{1}, Completion: []bool{false, true},
		}
	}
	example := Example{ID: reference.id, Group: reference.group, Values: []Value{value(RoleChosen), value(RoleRejected)}}
	if err := validateExample(example, reference, processor); err != nil {
		t.Fatal(err)
	}
	example.Values[0].Completion = nil
	if err := validateExample(example, reference, processor); err == nil {
		t.Fatal("accepted chosen value without completion mask")
	}
}
