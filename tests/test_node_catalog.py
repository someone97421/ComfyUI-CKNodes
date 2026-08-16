import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOCALE_PATH = ROOT / "locales" / "zh" / "nodeDefs.json"


def literal_mapping(tree, assignment_name):
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            try:
                key_value = ast.literal_eval(key)
            except (ValueError, TypeError):
                continue
            result[key_value] = value
    return result


def collect_catalog():
    node_ids = set()
    categories = []
    display_names = {}
    for path in ROOT.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        node_ids.update(literal_mapping(tree, "NODE_CLASS_MAPPINGS"))

        for key, value in literal_mapping(tree, "NODE_DISPLAY_NAME_MAPPINGS").items():
            try:
                display_names[key] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == "CATEGORY" for target in node.targets):
                    try:
                        categories.append(ast.literal_eval(node.value))
                    except (ValueError, TypeError):
                        pass
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "category":
                        try:
                            categories.append(ast.literal_eval(keyword.value))
                        except (ValueError, TypeError):
                            pass
    return node_ids, categories, display_names


def collect_literal_inputs_and_options():
    result = {}
    class_nodes = {}
    mapping_classes = {}

    for path in ROOT.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for node_id, class_value in literal_mapping(tree, "NODE_CLASS_MAPPINGS").items():
            if isinstance(class_value, ast.Name):
                class_nodes[node_id] = classes.get(class_value.id)
                mapping_classes[node_id] = class_value.id

    for node_id, class_node in class_nodes.items():
        inputs = set()
        options = {}
        if class_node is None:
            continue

        for method in class_node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if method.name == "INPUT_TYPES":
                for return_node in (node for node in ast.walk(method) if isinstance(node, ast.Return)):
                    outer = return_node.value
                    if not isinstance(outer, ast.Dict):
                        continue
                    for section_key, section_value in zip(outer.keys, outer.values):
                        try:
                            section_name = ast.literal_eval(section_key)
                        except (ValueError, TypeError):
                            continue
                        if section_name not in ("required", "optional") or not isinstance(section_value, ast.Dict):
                            continue
                        for input_key, input_value in zip(section_value.keys, section_value.values):
                            try:
                                input_name = ast.literal_eval(input_key)
                            except (ValueError, TypeError):
                                continue
                            inputs.add(input_name)
                            if isinstance(input_value, (ast.Tuple, ast.List)) and input_value.elts:
                                first = input_value.elts[0]
                                if isinstance(first, ast.List):
                                    try:
                                        options[input_name] = set(ast.literal_eval(first))
                                    except (ValueError, TypeError):
                                        pass

            if method.name == "define_schema":
                for call in (node for node in ast.walk(method) if isinstance(node, ast.Call)):
                    if not isinstance(call.func, ast.Attribute) or call.func.attr != "Input" or not call.args:
                        continue
                    try:
                        input_name = ast.literal_eval(call.args[0])
                    except (ValueError, TypeError):
                        continue
                    inputs.add(input_name)
                    for keyword in call.keywords:
                        if keyword.arg == "options":
                            try:
                                options[input_name] = set(ast.literal_eval(keyword.value))
                            except (ValueError, TypeError):
                                pass

        result[node_id] = {"inputs": inputs, "options": options, "class": mapping_classes.get(node_id)}
    return result


class NodeCatalogTest(unittest.TestCase):
    def test_frontend_directory_is_declared(self):
        init_text = (ROOT / "__init__.py").read_text(encoding="utf-8-sig")
        self.assertIn('WEB_DIRECTORY = "./web"', init_text)
        self.assertIn("'WEB_DIRECTORY'", init_text)

    def test_chinese_locale_is_valid_json(self):
        data = json.loads(LOCALE_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)

    def test_locale_covers_every_registered_node(self):
        node_ids, _, _ = collect_catalog()
        translations = json.loads(LOCALE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(node_ids, set(translations))

    def test_every_category_uses_unified_root(self):
        _, categories, _ = collect_catalog()
        self.assertTrue(categories)
        invalid = [category for category in categories if not category.startswith("CK Nodes/")]
        self.assertEqual(invalid, [])

    def test_default_display_names_are_english_and_unified(self):
        node_ids, _, display_names = collect_catalog()
        self.assertEqual(node_ids, set(display_names))
        invalid = [name for name in display_names.values() if not name.startswith("CK ")]
        self.assertEqual(invalid, [])

    def test_smart_merge_is_registered(self):
        node_ids, _, _ = collect_catalog()
        self.assertIn("CKSmartMergeImages", node_ids)

    def test_locale_covers_literal_input_names_and_options(self):
        translations = json.loads(LOCALE_PATH.read_text(encoding="utf-8"))
        catalog = collect_literal_inputs_and_options()
        missing_inputs = []
        missing_options = []
        for node_id, definition in catalog.items():
            translated_inputs = translations[node_id].get("inputs", {})
            for input_name in definition["inputs"]:
                if input_name not in translated_inputs:
                    missing_inputs.append((node_id, input_name))
            for input_name, option_values in definition["options"].items():
                translated_options = translated_inputs.get(input_name, {}).get("options", {})
                for option_value in option_values:
                    if option_value not in translated_options:
                        missing_options.append((node_id, input_name, option_value))
        self.assertEqual(missing_inputs, [])
        self.assertEqual(missing_options, [])


if __name__ == "__main__":
    unittest.main()
