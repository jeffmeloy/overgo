package sampling

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/binaryschema"
)

type schemaBuiltinRule struct {
	content string
	deps    []string
}

const schemaSpaceRule = `| " " | "\n"{1,2} [ \t]{0,20}`

var schemaPrimitiveRules = map[string]schemaBuiltinRule{
	"boolean":       {content: `("true" | "false")`},
	"decimal-part":  {content: `[0-9]{1,16}`},
	"integral-part": {content: `[0] | [1-9] [0-9]{0,15}`},
	"number": {
		content: `("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?`,
		deps:    []string{"integral-part", "decimal-part"},
	},
	"integer": {content: `("-"? integral-part)`, deps: []string{"integral-part"}},
	"value": {
		content: `object | array | string | number | boolean | null`,
		deps:    []string{"object", "array", "string", "number", "boolean", "null"},
	},
	"object": {
		content: `"{" space ( string ":" space value ("," space string ":" space value)* )? space "}"`,
		deps:    []string{"string", "value"},
	},
	"array":  {content: `"[" space ( value ("," space value)* )? space "]"`, deps: []string{"value"}},
	"uuid":   {content: `"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""`},
	"char":   {content: `[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`},
	"string": {content: `"\"" char* "\""`, deps: []string{"char"}},
	"null":   {content: `"null"`},
}

var schemaStringFormatRules = map[string]schemaBuiltinRule{
	"date":             {content: `[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )`},
	"time":             {content: `([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )`},
	"date-time":        {content: `date "T" time`, deps: []string{"date", "time"}},
	"date-string":      {content: `"\"" date "\""`, deps: []string{"date"}},
	"time-string":      {content: `"\"" time "\""`, deps: []string{"time"}},
	"date-time-string": {content: `"\"" date-time "\""`, deps: []string{"date-time"}},
}

var invalidSchemaRuleCharacters = regexp.MustCompile(`[^a-zA-Z0-9-]+`)

type schemaConverter struct {
	rules     map[string]string
	refs      map[string]any
	resolving map[string]bool
}

type schemaStringTrie struct {
	children map[rune]*schemaStringTrie
	terminal bool
}

// JSONSchemaToGrammar: converts pinned llama.cpp JSON-schema subset to
// deterministic GBNF while preserving input property order
func JSONSchemaToGrammar(input []byte) (string, error) {
	value, err := parseOrderedJSON(input)
	if err != nil {
		return "", fmt.Errorf("JSON schema: %w", err)
	}
	root, ok := value.(orderedJSONObject)
	if !ok {
		return "", errors.New("JSON schema root must be an object")
	}
	converter := &schemaConverter{
		rules:     map[string]string{"space": schemaSpaceRule},
		refs:      make(map[string]any),
		resolving: make(map[string]bool),
	}
	if err := converter.indexRefs(root); err != nil {
		return "", err
	}
	if _, err := converter.visit(root, ""); err != nil {
		return "", err
	}
	return converter.formatGrammar(), nil
}

func (converter *schemaConverter) addRule(name, rule string) string {
	name = invalidSchemaRuleCharacters.ReplaceAllString(name, "-")
	key := name
	if existing, ok := converter.rules[key]; ok && existing != rule {
		for suffix := 0; ; suffix++ {
			candidate := key + strconv.Itoa(suffix)
			existing, occupied := converter.rules[candidate]
			if !occupied || existing == rule {
				key = candidate
				break
			}
		}
	}
	converter.rules[key] = rule
	return key
}

func (converter *schemaConverter) addPrimitive(name string, rule schemaBuiltinRule) (string, error) {
	result := converter.addRule(name, rule.content)
	for _, dependency := range rule.deps {
		if _, exists := converter.rules[dependency]; exists {
			continue
		}
		dependencyRule, ok := schemaPrimitiveRules[dependency]
		if !ok {
			dependencyRule, ok = schemaStringFormatRules[dependency]
		}
		if !ok {
			return "", fmt.Errorf("unknown primitive rule %q", dependency)
		}
		if _, err := converter.addPrimitive(dependency, dependencyRule); err != nil {
			return "", err
		}
	}
	return result, nil
}

func schemaRuleName(name string) string {
	if name == "" {
		return "root"
	}
	if name == "root" ||
		name == "space" ||
		schemaPrimitiveRules[name].content != "" ||
		schemaStringFormatRules[name].content != "" {
		return name + "-"
	}
	return name
}

func schemaGrammarLiteral(value string) string {
	replacer := strings.NewReplacer(
		`\`, `\\`,
		`"`, `\"`,
		"\r", `\r`,
		"\n", `\n`,
	)
	return `"` + replacer.Replace(value) + `"`
}

func (converter *schemaConverter) constantRule(value any) (string, error) {
	encoded, err := marshalOrderedJSON(value)
	if err != nil {
		return "", err
	}
	return schemaGrammarLiteral(encoded), nil
}

func (converter *schemaConverter) notStrings(values []string) (string, error) {
	root := &schemaStringTrie{children: make(map[rune]*schemaStringTrie)}
	for _, value := range values {
		node := root
		for _, character := range value {
			child := node.children[character]
			if child == nil {
				child = &schemaStringTrie{
					children: make(map[rune]*schemaStringTrie),
				}
				node.children[character] = child
			}
			node = child
		}
		node.terminal = true
	}
	charRule, err := converter.addPrimitive("char", schemaPrimitiveRules["char"])
	if err != nil {
		return "", err
	}
	var output strings.Builder
	output.WriteString(`["] ( `)
	var visit func(*schemaStringTrie)
	visit = func(node *schemaStringTrie) {
		characters := make([]rune, 0, len(node.children))
		for character := range node.children {
			characters = append(characters, character)
		}
		sort.Slice(characters, func(left, right int) bool {
			return characters[left] < characters[right]
		})
		for index, character := range characters {
			if index > 0 {
				output.WriteString(" | ")
			}
			child := node.children[character]
			fmt.Fprintf(&output, "[%c]", character)
			if len(child.children) > 0 {
				output.WriteString(" (")
				visit(child)
				output.WriteByte(')')
			} else if child.terminal {
				output.WriteByte(' ')
				output.WriteString(charRule)
				output.WriteByte('+')
			}
		}
		if len(characters) > 0 {
			output.WriteString(` | [^"`)
			for _, character := range characters {
				output.WriteRune(character)
			}
			output.WriteString(`] `)
			output.WriteString(charRule)
			output.WriteByte('*')
		}
	}
	visit(root)
	output.WriteString(` )`)
	if !root.terminal {
		output.WriteByte('?')
	}
	output.WriteString(` ["]`)
	return output.String(), nil
}

func schemaString(object orderedJSONObject, name string) (string, bool) {
	value, ok := object.get(name)
	if !ok {
		return "", false
	}
	result, ok := value.(string)
	return result, ok
}

func schemaArray(object orderedJSONObject, name string) ([]any, bool) {
	value, ok := object.get(name)
	if !ok {
		return nil, false
	}
	result, ok := value.([]any)
	return result, ok
}

func (converter *schemaConverter) visit(schema orderedJSONObject, name string) (string, error) {
	schemaType, _ := schemaString(schema, "type")
	schemaFormat, _ := schemaString(schema, "format")
	ruleName := schemaRuleName(name)

	if ref, ok := schemaString(schema, "$ref"); ok {
		resolvedName, err := converter.resolveRef(ref)
		if err != nil {
			return "", err
		}
		return converter.addRule(ruleName, resolvedName), nil
	}
	for _, keyword := range []string{"oneOf", "anyOf"} {
		if alternatives, ok := schemaArray(schema, keyword); ok {
			rules := make([]string, len(alternatives))
			for index, alternative := range alternatives {
				object, ok := alternative.(orderedJSONObject)
				if !ok {
					return "", fmt.Errorf("%s alternative %d is not an object", keyword, index)
				}
				alternativeName := name + "-"
				if name == "" {
					alternativeName = "alternative-"
				}
				var err error
				rules[index], err = converter.visit(
					object,
					alternativeName+strconv.Itoa(index),
				)
				if err != nil {
					return "", err
				}
			}
			return converter.addRule(ruleName, strings.Join(rules, " | ")), nil
		}
	}
	if types, ok := schemaArray(schema, "type"); ok {
		rules := make([]string, len(types))
		for index, item := range types {
			itemType, ok := item.(string)
			if !ok {
				return "", fmt.Errorf("type alternative %d is not a string", index)
			}
			copySchema := schema
			copySchema.Members = slices.Clone(schema.Members)
			for memberIndex := range copySchema.Members {
				if copySchema.Members[memberIndex].Name == "type" {
					copySchema.Members[memberIndex].Value = itemType
				}
			}
			alternativeName := name + "-"
			if name == "" {
				alternativeName = "alternative-"
			}
			var err error
			rules[index], err = converter.visit(copySchema, alternativeName+strconv.Itoa(index))
			if err != nil {
				return "", err
			}
		}
		return converter.addRule(ruleName, strings.Join(rules, " | ")), nil
	}
	if value, ok := schema.get("const"); ok {
		rule, err := converter.constantRule(value)
		if err != nil {
			return "", err
		}
		return converter.addRule(ruleName, rule), nil
	}
	if values, ok := schemaArray(schema, "enum"); ok {
		rules := make([]string, len(values))
		for index, value := range values {
			var err error
			rules[index], err = converter.constantRule(value)
			if err != nil {
				return "", err
			}
		}
		return converter.addRule(ruleName, "("+strings.Join(rules, " | ")+")"), nil
	}
	if (schemaType == "" || schemaType == "object") &&
		(schema.has("properties") || schema.has("additionalProperties")) {
		return converter.visitObject(schema, name, ruleName)
	}
	if (schemaType == "" || schemaType == "object" || schemaType == "string") &&
		schema.has("allOf") {
		return converter.visitAllOf(schema, name, ruleName)
	}
	if (schemaType == "" || schemaType == "array") &&
		(schema.has("items") || schema.has("prefixItems")) {
		return converter.visitArray(schema, name, ruleName)
	}
	if (schemaType == "" || schemaType == "string") && schema.has("pattern") {
		pattern, ok := schemaString(schema, "pattern")
		if !ok {
			return "", errors.New("JSON schema pattern is not a string")
		}
		return converter.visitPattern(pattern, ruleName)
	}
	if (schemaType == "" || schemaType == "string") &&
		regexp.MustCompile(`^uuid[1-5]?$`).MatchString(schemaFormat) {
		primitiveName := schemaFormat
		if ruleName == "root" {
			primitiveName = "root"
		}
		return converter.addPrimitive(primitiveName, schemaPrimitiveRules["uuid"])
	}
	if (schemaType == "" || schemaType == "string") && schemaFormat != "" {
		primitiveName := schemaFormat + "-string"
		if primitive, ok := schemaStringFormatRules[primitiveName]; ok {
			added, err := converter.addPrimitive(primitiveName, primitive)
			if err != nil {
				return "", err
			}
			return converter.addRule(ruleName, added), nil
		}
	}
	if schemaType == "string" && (schema.has("minLength") || schema.has("maxLength")) {
		return converter.visitBoundedString(schema, ruleName)
	}
	if (schemaType == "" || schemaType == "integer") &&
		(schema.has("minimum") ||
			schema.has("exclusiveMinimum") ||
			schema.has("maximum") ||
			schema.has("exclusiveMaximum")) {
		var minimum, maximum *int64
		if schema.has("minimum") || schema.has("exclusiveMinimum") {
			name := "minimum"
			delta := int64(0)
			if !schema.has(name) {
				name = "exclusiveMinimum"
				delta = 1
			}
			value, err := schemaInt64(schema, name)
			if err != nil {
				return "", err
			}
			value += delta
			minimum = &value
		}
		if schema.has("maximum") || schema.has("exclusiveMaximum") {
			name := "maximum"
			delta := int64(0)
			if !schema.has(name) {
				name = "exclusiveMaximum"
				delta = -1
			}
			value, err := schemaInt64(schema, name)
			if err != nil {
				return "", err
			}
			value += delta
			maximum = &value
		}
		var rule string
		var err error
		switch {
		case minimum != nil && maximum != nil:
			rule, err = buildSchemaIntegerRange(*minimum, *maximum)
		case minimum != nil:
			rule, err = buildSchemaIntegerMinimum(*minimum)
		default:
			rule, err = buildSchemaIntegerMaximum(*maximum)
		}
		if err != nil {
			return "", err
		}
		return converter.addRule(ruleName, "("+rule+")"), nil
	}
	if schemaType == "object" || len(schema.Members) == 0 {
		primitive, err := converter.addPrimitive("object", schemaPrimitiveRules["object"])
		return converter.addRule(ruleName, primitive), err
	}
	if schemaType == "" {
		primitive, err := converter.addPrimitive("value", schemaPrimitiveRules["value"])
		return converter.addRule(ruleName, primitive), err
	}
	primitive, ok := schemaPrimitiveRules[schemaType]
	if !ok {
		return "", fmt.Errorf("unrecognized schema type %q", schemaType)
	}
	primitiveName := schemaType
	if ruleName == "root" {
		primitiveName = "root"
	}
	return converter.addPrimitive(primitiveName, primitive)
}

func (converter *schemaConverter) visitAllOf(
	schema orderedJSONObject,
	name, ruleName string,
) (string, error) {
	components, ok := schemaArray(schema, "allOf")
	if !ok {
		return "", errors.New("allOf is not an array")
	}
	properties := orderedJSONObject{}
	required := make(map[string]bool)
	var enumSets []map[string]any
	var addComponent func(any, bool) error
	addComponent = func(value any, isRequired bool) error {
		component, ok := value.(orderedJSONObject)
		if !ok {
			return errors.New("allOf component is not a schema object")
		}
		if ref, ok := schemaString(component, "$ref"); ok {
			resolved, exists := converter.refs[ref]
			if !exists {
				return fmt.Errorf("unresolved allOf ref %q", ref)
			}
			component, ok = resolved.(orderedJSONObject)
			if !ok {
				return fmt.Errorf("allOf ref %q is not an object schema", ref)
			}
		}
		if value, exists := component.get("properties"); exists {
			object, ok := value.(orderedJSONObject)
			if !ok {
				return errors.New("allOf properties is not an object")
			}
			for _, property := range object.Members {
				properties.Members = append(properties.Members, property)
				if isRequired {
					required[property.Name] = true
				}
			}
		}
		if values, exists := schemaArray(component, "enum"); exists {
			set := make(map[string]any, len(values))
			for _, item := range values {
				encoded, err := marshalOrderedJSON(item)
				if err != nil {
					return err
				}
				set[encoded] = item
			}
			enumSets = append(enumSets, set)
		}
		return nil
	}
	for _, component := range components {
		object, ok := component.(orderedJSONObject)
		if !ok {
			return "", errors.New("allOf component is not an object")
		}
		if alternatives, ok := schemaArray(object, "anyOf"); ok {
			for _, alternative := range alternatives {
				if err := addComponent(alternative, false); err != nil {
					return "", err
				}
			}
		} else if err := addComponent(object, true); err != nil {
			return "", err
		}
	}
	if len(enumSets) > 0 {
		intersection := enumSets[0]
		for _, set := range enumSets[1:] {
			maps.DeleteFunc(intersection, func(encoded string, _ any) bool {
				_, exists := set[encoded]
				return !exists
			})
		}
		if len(intersection) > 0 {
			encoded := make([]string, 0, len(intersection))
			for value := range intersection {
				encoded = append(encoded, value)
			}
			slices.Sort(encoded)
			rules := make([]string, len(encoded))
			for index, value := range encoded {
				rules[index] = schemaGrammarLiteral(value)
			}
			return converter.addRule(
				ruleName,
				"("+strings.Join(rules, " | ")+")",
			), nil
		}
	}
	requiredValues := make([]any, 0, len(required))
	for _, property := range properties.Members {
		if required[property.Name] {
			requiredValues = append(requiredValues, property.Name)
		}
	}
	combined := orderedJSONObject{Members: []orderedJSONMember{
		{Name: "type", Value: "object"},
		{Name: "properties", Value: properties},
		{Name: "required", Value: requiredValues},
	}}
	return converter.visitObject(combined, name, ruleName)
}

func (converter *schemaConverter) formatGrammar() string {
	names := make([]string, 0, len(converter.rules))
	for name := range converter.rules {
		names = append(names, name)
	}
	slices.Sort(names)
	lines := make([]string, len(names))
	for index, name := range names {
		lines[index] = name + " ::= " + converter.rules[name]
	}
	return strings.Join(lines, "\n")
}

func buildSchemaRepetition(item string, minimum int, maximum *int, separator string) string {
	if maximum != nil && *maximum == 0 {
		return ""
	}
	if minimum == 0 && maximum != nil && *maximum == 1 {
		return item + "?"
	}
	if separator == "" {
		if minimum == 1 && maximum == nil {
			return item + "+"
		}
		if minimum == 0 && maximum == nil {
			return item + "*"
		}
		maximumText := ""
		if maximum != nil {
			maximumText = strconv.Itoa(*maximum)
		}
		return fmt.Sprintf("%s{%d,%s}", item, minimum, maximumText)
	}
	nextMinimum := max(minimum-1, 0)
	var nextMaximum *int
	if maximum != nil {
		value := *maximum - 1
		nextMaximum = &value
	}
	result := item + " " + buildSchemaRepetition(
		"("+separator+" "+item+")",
		nextMinimum,
		nextMaximum,
		"",
	)
	if minimum == 0 {
		result = "(" + result + ")?"
	}
	return result
}

func schemaInteger(object orderedJSONObject, name string, fallback int) (int, error) {
	value, ok := object.get(name)
	if !ok {
		return fallback, nil
	}
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%s is not an integer", name)
	}
	parsed, err := strconv.Atoi(string(number))
	if err != nil || parsed < 0 {
		return 0, fmt.Errorf("%s is not a non-negative integer", name)
	}
	return parsed, nil
}

func schemaInt64(object orderedJSONObject, name string) (int64, error) {
	value, ok := object.get(name)
	if !ok {
		return 0, fmt.Errorf("%s is missing", name)
	}
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%s is not an integer", name)
	}
	result, err := strconv.ParseInt(string(number), binaryschema.DecimalRadix, binaryschema.Width64Bits)
	if err != nil {
		return 0, fmt.Errorf("%s is not an int64: %w", name, err)
	}
	return result, nil
}

func (converter *schemaConverter) indexRefs(root orderedJSONObject) error {
	converter.refs["#"] = root
	var visit func(any) error
	visit = func(value any) error {
		switch typed := value.(type) {
		case orderedJSONObject:
			if ref, ok := schemaString(typed, "$ref"); ok {
				if !strings.HasPrefix(ref, "#/") && ref != "#" {
					return fmt.Errorf("unsupported JSON schema ref %q", ref)
				}
				target := any(root)
				if ref != "#" {
					for _, encoded := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
						selector := strings.ReplaceAll(strings.ReplaceAll(encoded, "~1", "/"), "~0", "~")
						switch current := target.(type) {
						case orderedJSONObject:
							target, ok = current.get(selector)
							if !ok {
								return fmt.Errorf("ref %q has no component %q", ref, selector)
							}
						case []any:
							index, err := strconv.Atoi(selector)
							if err != nil || index < 0 || index >= len(current) {
								return fmt.Errorf("ref %q has no array index %q", ref, selector)
							}
							target = current[index]
						default:
							return fmt.Errorf("ref %q traverses scalar at %q", ref, selector)
						}
					}
				}
				converter.refs[ref] = target
			}
			for _, member := range typed.Members {
				if err := visit(member.Value); err != nil {
					return err
				}
			}
		case []any:
			for _, item := range typed {
				if err := visit(item); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return visit(root)
}

func (converter *schemaConverter) resolveRef(ref string) (string, error) {
	target, ok := converter.refs[ref]
	if !ok {
		return "", fmt.Errorf("unresolved JSON schema ref %q", ref)
	}
	name := "ref" + invalidSchemaRuleCharacters.ReplaceAllString(
		strings.SplitN(ref, "#", 2)[len(strings.SplitN(ref, "#", 2))-1],
		"-",
	)
	if name == "ref" {
		name = "ref-root"
	}
	if converter.resolving[ref] {
		return name, nil
	}
	if _, exists := converter.rules[name]; exists {
		return name, nil
	}
	object, ok := target.(orderedJSONObject)
	if !ok {
		return "", fmt.Errorf("ref %q does not resolve to an object schema", ref)
	}
	converter.resolving[ref] = true
	result, err := converter.visit(object, name)
	delete(converter.resolving, ref)
	return result, err
}

func (converter *schemaConverter) visitArray(
	schema orderedJSONObject,
	name, ruleName string,
) (string, error) {
	items, ok := schema.get("items")
	if !ok {
		items, _ = schema.get("prefixItems")
	}
	if tuple, ok := items.([]any); ok {
		rules := make([]string, len(tuple))
		for index, item := range tuple {
			object, ok := item.(orderedJSONObject)
			if !ok {
				return "", fmt.Errorf("tuple item %d is not a schema object", index)
			}
			itemName := name
			if itemName != "" {
				itemName += "-"
			}
			var err error
			rules[index], err = converter.visit(object, itemName+"tuple-"+strconv.Itoa(index))
			if err != nil {
				return "", err
			}
		}
		return converter.addRule(
			ruleName,
			`"[" space `+strings.Join(rules, ` "," space `)+` space "]"`,
		), nil
	}
	object, ok := items.(orderedJSONObject)
	if !ok {
		return "", errors.New("array items is not a schema object")
	}
	itemName := name
	if itemName != "" {
		itemName += "-"
	}
	itemRule, err := converter.visit(object, itemName+"item")
	if err != nil {
		return "", err
	}
	minimum, err := schemaInteger(schema, "minItems", 0)
	if err != nil {
		return "", err
	}
	var maximum *int
	if schema.has("maxItems") {
		value, valueErr := schemaInteger(schema, "maxItems", 0)
		if valueErr != nil {
			return "", valueErr
		}
		maximum = &value
	}
	return converter.addRule(
		ruleName,
		`"[" space `+buildSchemaRepetition(itemRule, minimum, maximum, `"," space`)+` space "]"`,
	), nil
}

func (converter *schemaConverter) visitBoundedString(
	schema orderedJSONObject,
	ruleName string,
) (string, error) {
	charRule, err := converter.addPrimitive("char", schemaPrimitiveRules["char"])
	if err != nil {
		return "", err
	}
	minimum, err := schemaInteger(schema, "minLength", 0)
	if err != nil {
		return "", err
	}
	var maximum *int
	if schema.has("maxLength") {
		value, valueErr := schemaInteger(schema, "maxLength", 0)
		if valueErr != nil {
			return "", valueErr
		}
		maximum = &value
	}
	return converter.addRule(
		ruleName,
		`"\"" `+buildSchemaRepetition(charRule, minimum, maximum, "")+` "\""`,
	), nil
}

func (converter *schemaConverter) visitObject(
	schema orderedJSONObject,
	name, ruleName string,
) (string, error) {
	properties := orderedJSONObject{}
	if value, ok := schema.get("properties"); ok {
		var valid bool
		properties, valid = value.(orderedJSONObject)
		if !valid {
			return "", errors.New("properties is not an object")
		}
	}
	required := make(map[string]bool)
	if values, ok := schemaArray(schema, "required"); ok {
		for index, value := range values {
			property, ok := value.(string)
			if !ok {
				return "", fmt.Errorf("required item %d is not a string", index)
			}
			required[property] = true
		}
	}
	propertyRules := make(map[string]string, len(properties.Members)+1)
	requiredNames := make([]string, 0)
	optionalNames := make([]string, 0)
	for _, property := range properties.Members {
		propertySchema, ok := property.Value.(orderedJSONObject)
		if !ok {
			return "", fmt.Errorf("property %q is not a schema object", property.Name)
		}
		prefix := name
		if prefix != "" {
			prefix += "-"
		}
		valueRule, err := converter.visit(propertySchema, prefix+property.Name)
		if err != nil {
			return "", err
		}
		keyJSON, _ := json.Marshal(property.Name)
		propertyRules[property.Name] = converter.addRule(
			prefix+property.Name+"-kv",
			schemaGrammarLiteral(string(keyJSON))+` space ":" space `+valueRule,
		)
		if required[property.Name] {
			requiredNames = append(requiredNames, property.Name)
		} else {
			optionalNames = append(optionalNames, property.Name)
		}
	}
	additional, hasAdditional := schema.get("additionalProperties")
	if len(properties.Members) == 0 && hasAdditional && additional == true {
		primitive, err := converter.addPrimitive("object", schemaPrimitiveRules["object"])
		if err != nil {
			return "", err
		}
		return converter.addRule(ruleName, primitive), nil
	}
	if hasAdditional && additional != false {
		prefix := name
		if prefix != "" {
			prefix += "-"
		}
		var valueRule string
		var err error
		if object, ok := additional.(orderedJSONObject); ok {
			valueRule, err = converter.visit(object, prefix+"additional-value")
		} else {
			valueRule, err = converter.addPrimitive("value", schemaPrimitiveRules["value"])
		}
		if err != nil {
			return "", err
		}
		var keyRule string
		if len(properties.Members) == 0 {
			keyRule, err = converter.addPrimitive("string", schemaPrimitiveRules["string"])
		} else {
			var rule string
			rule, err = converter.notStrings(properties.keys())
			if err == nil {
				keyRule = converter.addRule(prefix+"additional-k", rule)
			}
		}
		if err != nil {
			return "", err
		}
		propertyRules["*"] = converter.addRule(
			prefix+"additional-kv",
			keyRule+` ":" space `+valueRule,
		)
		optionalNames = append(optionalNames, "*")
	}
	rule := `"{" space ` + joinSchemaPropertyRules(propertyRules, requiredNames, optionalNames, converter, name) + ` space "}"`
	return converter.addRule(ruleName, rule), nil
}

func joinSchemaPropertyRules(
	rules map[string]string,
	required, optional []string,
	converter *schemaConverter,
	name string,
) string {
	parts := make([]string, len(required))
	for index, property := range required {
		parts[index] = rules[property]
	}
	result := strings.Join(parts, ` "," space `)
	if len(optional) == 0 {
		return result
	}
	prefix := name
	if prefix != "" {
		prefix += "-"
	}
	var recursive func([]string, bool) string
	recursive = func(properties []string, firstOptional bool) string {
		property := properties[0]
		keyValue := rules[property]
		comma := `( "," space ` + keyValue + ` )`
		var value string
		if firstOptional {
			value = comma
			if property == "*" {
				value += "*"
			} else {
				value += "?"
			}
		} else {
			value = keyValue
			if property == "*" {
				value += " " + comma + "*"
			}
		}
		if len(properties) > 1 {
			rest := converter.addRule(
				prefix+property+"-rest",
				recursive(properties[1:], true),
			)
			value += " " + rest
		}
		return value
	}
	alternatives := make([]string, len(optional))
	for index := range optional {
		alternatives[index] = recursive(optional[index:], false)
	}
	alternativesRule := strings.Join(alternatives, " | ")
	if result != "" {
		return result + ` ( "," space ( ` + alternativesRule + ` ) )?`
	}
	return ` (` + alternativesRule + ` )?`
}
