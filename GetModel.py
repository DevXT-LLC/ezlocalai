import argparse
from provider import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", type=str, default="None")
    args = parser.parse_args()
    model_url = args.model_url
    if model_url != "None":
        model_path = get_model(model_url)
        print(model_path)
