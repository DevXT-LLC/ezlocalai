import sys
import os
import requests
import argparse


def get_model(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF", quant_type="Q4_K_M"):
    model_name = model_url.split("/")[-1].replace("-GGUF", "").lower()
    file_path = f"models/{model_name}.{quant_type}.gguf"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(file_path):
        url = (
            model_url
            if "https://" in model_url
            else f"https://huggingface.co/{model_url}/resolve/main/{model_name}.{quant_type}.gguf"
        )
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    if not os.path.exists(f"models/{model_name}.README.md"):
        readme_url = f"https://huggingface.co/{model_url}/raw/main/README.md"
        with requests.get(readme_url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(f"models/{model_name}.README.md", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    with open(f"models/{model_name}.README.md", "r") as f:
        readme = f.read()
    if not os.path.exists(f"models/{model_name}.txt"):
        prompt_template = readme.split("prompt_template: '")[1].split("'")[0]
        with open(f"models/{model_name}.txt", "w") as f:
            f.write(prompt_template)
    with open(f"models/prompt.txt", "w") as f:
        with open(f"models/{model_name}.txt", "r") as g:
            f.write(g.read())
    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", type=str, default="")
    parser.add_argument("--quant_type", type=str, default="Q4_K_M")
    args = parser.parse_args()
    model_url = args.model_url
    quant_type = args.quant_type
    if model_url != "":
        model_path = get_model(model_url, quant_type)
        print(model_path)
