package codemanifest

import (
	"cmp"
	"errors"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"path"
	"slices"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

const artifactImportPath = "overgo/internal/artifact"

var documentContractFields = [...]string{"Kind", "MediaType", "Schema"}
var jsonContractFields = [...]string{"Kind", "Schema"}

// DocumentDeclaration binds one static contract to source authority.
type DocumentDeclaration struct {
	Name           string
	Owner          string
	VersionOwner   string
	Kind           artifact.Kind
	MediaType      string
	Schema         string
	BuildContexts  []string
	Source         string
	SourceIdentity string
}

type constantBinding struct {
	expression  ast.Expr
	packagePath string
	aliases     map[string]string
	iotaValue   int64
}

type contractSource struct {
	file        repoanalysis.GoFile
	packagePath string
	aliases     map[string]string
	contexts    []string
}

// DocumentDeclarations compiles package-level artifact contracts.
func DocumentDeclarations(snapshot repoanalysis.SourceSnapshot, selections []gosource.BuildSelection) ([]DocumentDeclaration, error) {
	sources, err := contractSources(snapshot, selections)
	if err != nil {
		return nil, err
	}
	constants, err := packageConstants(sources)
	if err != nil {
		return nil, err
	}
	var declarations []DocumentDeclaration
	for _, source := range sources {
		syntax, err := source.file.Syntax()
		if err != nil {
			return nil, fmt.Errorf("code manifest: parse contracts in %s: %w", source.file.Path, err)
		}
		for _, declaration := range syntax.Decls {
			general, ok := declaration.(*ast.GenDecl)
			if !ok || general.Tok != token.VAR {
				continue
			}
			for _, item := range general.Specs {
				values := item.(*ast.ValueSpec)
				for index, expression := range values.Values {
					if index >= len(values.Names) {
						break
					}
					contract, name, found, err := staticContract(expression, source, constants)
					if err != nil {
						return nil, fmt.Errorf("code manifest: %s.%s: %w", source.packagePath, values.Names[index].Name, err)
					}
					if !found {
						continue
					}
					owner := source.packagePath + "." + values.Names[index].Name
					if name == "" {
						name = values.Names[index].Name
					}
					declarations = append(declarations, DocumentDeclaration{
						Name: name, Owner: owner, VersionOwner: owner, Kind: contract.Kind,
						MediaType: contract.MediaType, Schema: contract.Schema,
						BuildContexts: slices.Clone(source.contexts), Source: source.file.Path,
						SourceIdentity: source.file.ContentID,
					})
				}
			}
		}
	}
	slices.SortFunc(declarations, func(left, right DocumentDeclaration) int {
		return cmp.Or(cmp.Compare(left.Owner, right.Owner), cmp.Compare(left.Schema, right.Schema))
	})
	return declarations, nil
}

func contractSources(snapshot repoanalysis.SourceSnapshot, selections []gosource.BuildSelection) ([]contractSource, error) {
	if len(selections) == 0 {
		return nil, errors.New("code manifest: document census has no build selection")
	}
	var sources []contractSource
	for _, file := range snapshot.Files {
		if file.Test {
			continue
		}
		var contexts []string
		packagePath := ""
		for _, selection := range selections {
			if !selection.Files[file.Path] {
				continue
			}
			contexts = append(contexts, selection.Context)
			selectedPackage := selection.Packages[file.Path]
			if selectedPackage == "" {
				selectedPackage = "overgo/" + path.Dir(file.Path)
			}
			if packagePath != "" && packagePath != selectedPackage {
				return nil, fmt.Errorf("code manifest: inconsistent package for %s", file.Path)
			}
			packagePath = selectedPackage
		}
		if len(contexts) == 0 {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			return nil, err
		}
		aliases := make(map[string]string, len(syntax.Imports))
		for _, imported := range syntax.Imports {
			importPath, err := strconv.Unquote(imported.Path.Value)
			if err != nil {
				return nil, err
			}
			alias := path.Base(importPath)
			if imported.Name != nil {
				alias = imported.Name.Name
			}
			aliases[alias] = importPath
		}
		slices.Sort(contexts)
		sources = append(sources, contractSource{file: file, packagePath: packagePath, aliases: aliases, contexts: slices.Compact(contexts)})
	}
	return sources, nil
}

func packageConstants(sources []contractSource) (map[string]constantBinding, error) {
	bindings := map[string]constantBinding{}
	for _, source := range sources {
		syntax, err := source.file.Syntax()
		if err != nil {
			return nil, err
		}
		for _, declaration := range syntax.Decls {
			general, ok := declaration.(*ast.GenDecl)
			if !ok || general.Tok != token.CONST && general.Tok != token.VAR {
				continue
			}
			var inherited []ast.Expr
			for ordinal, item := range general.Specs {
				values := item.(*ast.ValueSpec)
				if len(values.Values) != 0 {
					inherited = values.Values
				}
				if general.Tok == token.VAR && len(values.Values) == 0 {
					continue
				}
				for index, name := range values.Names {
					if len(inherited) == 0 {
						continue
					}
					expression := inherited[min(index, len(inherited)-1)]
					bindings[source.packagePath+"\x00"+name.Name] = constantBinding{
						expression: expression, packagePath: source.packagePath,
						aliases: source.aliases, iotaValue: int64(ordinal),
					}
				}
			}
		}
	}
	return bindings, nil
}

func staticContract(expression ast.Expr, source contractSource, constants map[string]constantBinding) (artifact.DocumentContract, string, bool, error) {
	switch value := expression.(type) {
	case *ast.CallExpr:
		owner, name := selectedCall(value.Fun, source.aliases)
		switch {
		case owner == artifactImportPath && name == "JSONDocumentCodec":
			if len(value.Args[1:]) < len(documentContractFields) {
				return artifact.DocumentContract{}, "", true, errors.New("incomplete JSON document codec")
			}
			if sharedContractFields(value.Args[1:]) {
				return artifact.DocumentContract{}, "", false, nil
			}
			label, ok := stringValue(value.Args[0], source.packagePath, source.aliases, constants, nil)
			if !ok {
				return artifact.DocumentContract{}, "", true, errors.New("dynamic document name")
			}
			contract, err := contractValues(value.Args[1], value.Args[2], value.Args[3], source, constants)
			return contract, label, true, err
		case owner == artifactImportPath && name == "JSONContract":
			if len(value.Args) != len(jsonContractFields) {
				return artifact.DocumentContract{}, "", true, errors.New("invalid JSON contract")
			}
			contract, err := contractValues(value.Args[0], nil, value.Args[1], source, constants)
			contract.MediaType = artifact.JSONMediaType
			return contract, "", true, err
		}
	case *ast.CompositeLit:
		owner, name := selectedType(value.Type, source.aliases)
		if owner == artifactImportPath && name == "DocumentContract" {
			contract, err := compositeContract(value, source, constants)
			return contract, "", true, err
		}
		if owner == artifactImportPath && name == "DocumentCodec" {
			for _, element := range value.Elts {
				field, ok := element.(*ast.KeyValueExpr)
				identifier, named := field.Key.(*ast.Ident)
				if ok && named && identifier.Name == "Contract" {
					if _, referenced := field.Value.(*ast.Ident); referenced {
						return artifact.DocumentContract{}, "", false, nil
					}
					contract, _, found, err := staticContract(field.Value, source, constants)
					return contract, "", found, err
				}
			}
		}
	case *ast.Ident:
		binding, found := constants[source.packagePath+"\x00"+value.Name]
		if found {
			return staticContract(binding.expression, source, constants)
		}
	}
	return artifact.DocumentContract{}, "", false, nil
}

func sharedContractFields(expressions []ast.Expr) bool {
	if len(expressions) < len(documentContractFields) {
		return false
	}
	expressions = expressions[:len(documentContractFields)]
	owner := ""
	for index, expression := range expressions {
		selector, ok := expression.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		identifier, named := selector.X.(*ast.Ident)
		if !named || selector.Sel.Name != documentContractFields[index] || owner != "" && owner != identifier.Name {
			return false
		}
		owner = identifier.Name
	}
	return owner != ""
}

func compositeContract(value *ast.CompositeLit, source contractSource, constants map[string]constantBinding) (artifact.DocumentContract, error) {
	fields := map[string]ast.Expr{}
	for _, element := range value.Elts {
		field, ok := element.(*ast.KeyValueExpr)
		identifier, named := field.Key.(*ast.Ident)
		if ok && named {
			fields[identifier.Name] = field.Value
		}
	}
	return contractValues(fields["Kind"], fields["MediaType"], fields["Schema"], source, constants)
}

func contractValues(kindExpression, mediaExpression, schemaExpression ast.Expr, source contractSource, constants map[string]constantBinding) (artifact.DocumentContract, error) {
	kindValue, ok := constantValue(kindExpression, source.packagePath, source.aliases, constants, map[string]bool{})
	if !ok {
		return artifact.DocumentContract{}, errors.New("dynamic artifact kind")
	}
	kindNumber, exact := constant.Uint64Val(constant.ToInt(kindValue))
	kind := artifact.Kind(kindNumber)
	if !exact || kind == artifact.KindInvalid || kind.String() == artifact.KindInvalid.String() {
		return artifact.DocumentContract{}, errors.New("invalid artifact kind")
	}
	mediaType := ""
	if mediaExpression != nil {
		mediaType, ok = stringValue(mediaExpression, source.packagePath, source.aliases, constants, nil)
		if !ok {
			return artifact.DocumentContract{}, errors.New("dynamic media type")
		}
	}
	schema, ok := stringValue(schemaExpression, source.packagePath, source.aliases, constants, nil)
	if !ok {
		return artifact.DocumentContract{}, errors.New("dynamic schema")
	}
	contract := artifact.DocumentContract{Kind: kind, MediaType: mediaType, Schema: schema}
	if mediaType != "" {
		if err := contract.Validate(); err != nil {
			return artifact.DocumentContract{}, err
		}
	}
	return contract, nil
}

func stringValue(expression ast.Expr, packagePath string, aliases map[string]string, constants map[string]constantBinding, visiting map[string]bool) (string, bool) {
	if visiting == nil {
		visiting = map[string]bool{}
	}
	value, ok := constantValue(expression, packagePath, aliases, constants, visiting)
	if !ok || value.Kind() != constant.String {
		return "", false
	}
	return constant.StringVal(value), true
}

func constantValue(expression ast.Expr, packagePath string, aliases map[string]string, constants map[string]constantBinding, visiting map[string]bool) (constant.Value, bool) {
	if expression == nil {
		return nil, false
	}
	switch value := expression.(type) {
	case *ast.BasicLit:
		result := constant.MakeFromLiteral(value.Value, value.Kind, uint(token.NoPos))
		return result, result.Kind() != constant.Unknown
	case *ast.ParenExpr:
		return constantValue(value.X, packagePath, aliases, constants, visiting)
	case *ast.Ident:
		key := packagePath + "\x00" + value.Name
		return boundConstant(key, constants, visiting)
	case *ast.SelectorExpr:
		qualifier, ok := value.X.(*ast.Ident)
		if !ok {
			return nil, false
		}
		if imported := aliases[qualifier.Name]; imported != "" {
			return boundConstant(imported+"\x00"+value.Sel.Name, constants, visiting)
		}
		binding, found := constants[packagePath+"\x00"+qualifier.Name]
		if !found {
			return nil, false
		}
		composite, ok := binding.expression.(*ast.CompositeLit)
		if !ok {
			return nil, false
		}
		for _, element := range composite.Elts {
			field, ok := element.(*ast.KeyValueExpr)
			identifier, named := field.Key.(*ast.Ident)
			if ok && named && identifier.Name == value.Sel.Name {
				return constantValue(field.Value, binding.packagePath, binding.aliases, constants, visiting)
			}
		}
		return nil, false
	case *ast.UnaryExpr:
		operand, ok := constantValue(value.X, packagePath, aliases, constants, visiting)
		if !ok {
			return nil, false
		}
		return constant.UnaryOp(value.Op, operand, uint(token.NoPos)), true
	case *ast.BinaryExpr:
		left, ok := constantValue(value.X, packagePath, aliases, constants, visiting)
		if !ok {
			return nil, false
		}
		right, ok := constantValue(value.Y, packagePath, aliases, constants, visiting)
		if !ok {
			return nil, false
		}
		return constant.BinaryOp(left, value.Op, right), true
	}
	return nil, false
}

func boundConstant(key string, constants map[string]constantBinding, visiting map[string]bool) (constant.Value, bool) {
	if visiting[key] {
		return nil, false
	}
	binding, found := constants[key]
	if !found {
		return nil, false
	}
	visiting[key] = true
	if identifier, ok := binding.expression.(*ast.Ident); ok && identifier.Name == "iota" {
		delete(visiting, key)
		return constant.MakeInt64(binding.iotaValue), true
	}
	value, ok := constantValue(binding.expression, binding.packagePath, binding.aliases, constants, visiting)
	delete(visiting, key)
	return value, ok
}

func selectedCall(expression ast.Expr, aliases map[string]string) (string, string) {
	expression = genericBase(expression)
	selector, ok := expression.(*ast.SelectorExpr)
	if !ok {
		return "", ""
	}
	qualifier, ok := selector.X.(*ast.Ident)
	if !ok {
		return "", ""
	}
	return aliases[qualifier.Name], selector.Sel.Name
}

func selectedType(expression ast.Expr, aliases map[string]string) (string, string) {
	return selectedCall(genericBase(expression), aliases)
}

func genericBase(expression ast.Expr) ast.Expr {
	for {
		switch value := expression.(type) {
		case *ast.IndexExpr:
			expression = value.X
		case *ast.IndexListExpr:
			expression = value.X
		default:
			return expression
		}
	}
}
