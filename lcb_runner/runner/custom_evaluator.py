import os
import json
import re
import ast

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    sort_and_extract_save_results,
    get_metrics,
)


def clean_code_output(code):
    """
    Cleans model-generated code before evaluation:
    - Extracts Python code from markdown backticks (` ```python ... ``` `).
    - Strips leading/trailing whitespace.
    - Ensures consistent indentation.
    """
    # Extract only the Python code from triple backticks
    match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
    if match:
        code = match.group(1)  # Extract content inside backticks

    # Strip unnecessary spaces
    code = code.strip()

    return code


def is_valid_python(code):
    """
    Checks if the given code is valid Python syntax.
    Returns True if valid, otherwise False.
    """
    try:
        ast.parse(code)  # Attempt to parse code
        return True
    except SyntaxError:
        return False


def main():
    args = get_args()

    benchmark, _ = build_prompt_benchmark(args)

    with open(args.custom_output_file, "r") as f:
        custom_outputs = json.load(f)
        assert isinstance(custom_outputs, list)

        if isinstance(custom_outputs[0], list):
            # Ensure the extracted outputs are properly formatted
            assert all(isinstance(custom_output, list) for custom_output in custom_outputs)

        elif isinstance(custom_outputs[0], dict):
            assert all(isinstance(custom_output, dict) for custom_output in custom_outputs)

            if args.scenario in [Scenario.codegeneration, Scenario.selfrepair]:
                # Extract and clean model outputs
                custom_outputs = [
                    [clean_code_output(output) for output in custom_output["code_list"]]
                    for custom_output in sorted(custom_outputs, key=lambda x: str(x["question_id"]))
                ]

            elif args.scenario == Scenario.testoutputprediction:
                custom_outputs = [
                    [clean_code_output(output) for output in custom_output["pred_list"]]
                    for custom_output in sorted(custom_outputs, key=lambda x: (str(x["question_id"]), str(x['test_id'])))
                ]

            elif args.scenario == Scenario.codeexecution:
                custom_outputs = [
                    [clean_code_output(output) for output in custom_output["pred_list"]]
                    for custom_output in sorted(custom_outputs, key=lambda x: int(x["id"].split("_")[1]))
                ]

    # Validate and filter out invalid Python code before testing
    custom_outputs = [
        [output for output in outputs_list if is_valid_python(output)]
        for outputs_list in custom_outputs
    ]

    save_results = [
        instance.insert_output(custom_output, custom_output)
        for instance, custom_output in zip(benchmark, custom_outputs)
    ]

    save_results, combined_results = sort_and_extract_save_results(args.scenario, save_results)

    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])

    if args.scenario == Scenario.codegeneration:
        metadatas = metrics[2]
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list, metadata=meta
            )
            for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                benchmark, combined_results, graded, metadatas
            )
        ]
    else:
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list
            )
            for instance, (outputs_list, extracted_list), graded_list in zip(
                benchmark, combined_results, graded
            )
        ]

    # Generate output paths
    if args.custom_output_save_name is None:
        output_path = args.custom_output_file[:-5] + f"_{args.scenario.value}_output.json"
    else:
        output_path = get_output_path(args.custom_output_save_name, args)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    with open(output_path.replace(".json", "_eval.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_path.replace(".json", "_eval_all.json"), "w") as f:
        json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()
